#### Preamble ####
# Purpose: Shiny App map visualization of shelter and neighbourhood time series forecasts using Random Forest 

# Load libraries
library(shiny)
library(leaflet)
library(sf)
library(tidyverse)
library(lubridate)
library(scales)

# Get data
shelters <- read_csv("..\\outputs\\shelter_rf.csv")
neighbourhoods <- read_csv("..\\outputs\\neighbourhood_rf.csv")
neighbourhood_map <- st_read("..\\data\\Neighbourhoods - 4326.shp")
shelter_occupancy_2023 <- read_csv("..\\data\\daily_shelter_overnight_occupancy.csv")
shelter_occupancy_2022 <- read_csv("..\\data\\daily-shelter-overnight-service-occupancy-capacity-2022.csv")
shelter_occupancy_2021 <- read_csv("..\\data\\daily-shelter-overnight-service-occupancy-capacity-2021.csv")
shelter_neighbourhood_data <- read_csv("..\\data\\shelter_neighbourhood_features_pca.csv")

# fix date format
shelter_occupancy_2021 <- shelter_occupancy_2021 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%y-%m-%d"))
shelter_occupancy_2022 <- shelter_occupancy_2022 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%y-%m-%d"))
shelter_occupancy_2023 <- shelter_occupancy_2023 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%Y-%m-%d"))
# combine data
shelter_occupancy <- rbind(shelter_occupancy_2021, shelter_occupancy_2022, shelter_occupancy_2023)


shelter_data <- unique(shelter_occupancy %>% select(LOCATION_ID, LOCATION_NAME, LOCATION_ADDRESS, LOCATION_POSTAL_CODE, LOCATION_CITY))

#### Prepare data ####

# create dataframe for neighbourhoods with shelters
neigh_has_shelter_df <- data.frame(
  `Neighbourhood` = unique(neighbourhoods$Neighbourhood),
  `has_shelter` = 'Has Shelter'
)

neighbourhood_geometry <- neighbourhood_map %>%
  select(AREA_NA7, geometry) %>%
  rename(Neighbourhood = AREA_NA7) %>%
  mutate(`Neighbourhood` = str_replace(`Neighbourhood`, 'North St.James Town', 'North St. James Town'))

neighbourhood_geometry <- left_join(neighbourhood_geometry, neigh_has_shelter_df) %>%
  mutate(`has_shelter` = replace_na(has_shelter, "No Shelter"))

neighbourhood_joined <- left_join(
  x=neighbourhoods,y=neighbourhood_geometry
) %>%
  mutate(date = make_date(year = year, month = month, day = day))

shelter_cleaned <- shelters %>%
  mutate(date = make_date(year = year, month = month, day = day))

individual_shelters <- unique(inner_join(shelter_data, shelter_cleaned) %>% select(LOCATION_ID, LOCATION_NAME, LAT, LON))

shelter_cleaned <- left_join(shelter_cleaned, shelter_data)

shelter_neighbourhoods <- unique(shelter_neighbourhood_data %>% select(LOCATION_NAME, Neighbourhood))

# prepare data for time series plots
shelter_occ_2122 <- rbind(shelter_occupancy_2021, shelter_occupancy_2022) %>%
  select(LOCATION_NAME, OCCUPANCY_DATE, SERVICE_USER_COUNT, CAPACITY_ACTUAL_ROOM, CAPACITY_ACTUAL_BED) %>%
  rename(date = OCCUPANCY_DATE) %>%
  mutate(SERVICE_USER_COUNT_PRED = NA)

shelter_time_plot <- rbind(
  shelter_cleaned %>% 
    select(LOCATION_NAME, date, SERVICE_USER_COUNT, SERVICE_USER_COUNT_PRED) %>%
    mutate(CAPACITY_ACTUAL_ROOM = NA, CAPACITY_ACTUAL_BED = NA),
  shelter_occ_2122
)

neighbourhood_time_plot <- left_join(shelter_time_plot, shelter_neighbourhoods) %>%
  group_by(Neighbourhood, date) %>%
  summarise(
    SERVICE_USER_COUNT = sum(SERVICE_USER_COUNT),
    SERVICE_USER_COUNT_PRED = sum(SERVICE_USER_COUNT_PRED)
  )

# function to make plot for shelters
makeTimeSeriesPlot <- function (loc_name) {
  selected_shelter_data <- shelter_time_plot[shelter_time_plot$LOCATION_NAME == loc_name,]
  selected_shelter_data <- selected_shelter_data[!is.na(selected_shelter_data$LOCATION_NAME),]
  
  selected_shelter_data <- selected_shelter_data %>%
    select(date, SERVICE_USER_COUNT, SERVICE_USER_COUNT_PRED) %>%
    group_by(date) %>%
    summarise(
      SERVICE_USER_COUNT = sum(SERVICE_USER_COUNT),
      SERVICE_USER_COUNT_PRED = sum(SERVICE_USER_COUNT_PRED)
    )
  
  colors <- c("Predicted" = "purple", "Actual" = "darkgreen")
  
  
  loc_cap <- shelter_occupancy[shelter_occupancy$LOCATION_NAME == loc_name,]
  loc_cap <- loc_cap[!is.na(loc_cap$LOCATION_NAME),]
  
  
  p <- ggplot() +
    geom_line(data = selected_shelter_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT, colour = "Actual")) +
    geom_line(data = selected_shelter_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT_PRED, colour = "Predicted")) +
    labs(title = str_wrap(str_c("Service users for ", loc_name), width = 50), x = "Date", y = "Service User Count", colour = "Legend") +
    scale_color_manual(values = colors) +
    scale_x_date(labels = date_format("%Y-%m")) +
    theme(legend.position = "top",
          legend.direction = "horizontal",
          legend.title = element_blank()
    ) +
    geom_vline(xintercept=as.Date("2023-01-01", format = "%Y-%m-%d"), colour = "red", linetype = 2)
  
  return (p)
}

# function to make plot for neighbourhoods
makeTimeSeriesPlotN <- function (loc_name) {
  selected_data <- neighbourhood_time_plot[neighbourhood_time_plot$Neighbourhood == loc_name,]
  selected_data <- selected_data[!is.na(selected_data$Neighbourhood),]
  
  colors <- c("Predicted" = "purple", "Actual" = "darkgreen")
  
  
  p <- ggplot() +
    geom_line(data = selected_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT, colour = "Actual")) +
    geom_line(data = selected_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT_PRED, colour = "Predicted")) +
    labs(title = str_wrap(str_c("Service users for ", loc_name), width = 50), x = "Date", y = "Service User Count", colour = "Legend") +
    scale_color_manual(values = colors) +
    scale_x_date(labels = date_format("%Y-%m")) +
    theme(legend.position = "top",
          legend.direction = "horizontal",
          legend.title = element_blank()
    ) +
    geom_vline(xintercept=as.Date("2023-01-01", format = "%Y-%m-%d"), colour = "red", linetype = 2)
  
  return (p)
}

#### Create shiny app ####

ui <- fluidPage(
  # App title
  titlePanel("Homeless Shelter Occupancy Predictions in Toronto"),
  
  sidebarLayout(
    # Main panel for displaying map
    mainPanel(
      tags$style(type = "text/css",
                 ".shiny-output-error { visibility: hidden; }",
                 ".shiny-output-error:before { visibility: hidden; }",
                 "#map {height: calc(100vh - 80px) !important;}"
      ),
      # Output: interactive map
      leafletOutput("map")
    ),
    
    # Sidebar panel 
    sidebarPanel(
      plotOutput("pred_model")
    )
  )
  
)

# colours for legend
fpal <- colorFactor(palette=c("#4DAF4A", "#E41A1C"), neighbourhood_geometry$has_shelter)

server <- function(input, output, session) {
  output$map <- renderLeaflet({
    leaflet() %>% addTiles() %>%
      addPolygons(
        data = neighbourhood_geometry,
        color = "black",
        popup = ~Neighbourhood,
        opacity = 1,
        weight = 1,
        fillOpacity = 0.6,
        layerId = ~Neighbourhood,
        fillColor = ~fpal(has_shelter),
        highlightOptions = highlightOptions(color = "white", weight = 2,
                                            bringToFront = TRUE)
      ) %>%
      addMarkers(
        data = individual_shelters,
        lng = ~LON,
        lat = ~LAT,
        popup = ~LOCATION_NAME,
        layerId = ~LOCATION_ID,
      ) %>%
      addLegend(
        data = neighbourhood_geometry,
        position = "bottomright",
        pal = fpal,
        values = ~has_shelter,
        title = "Neighbourhood",
        opacity = 0.6
      )
  })
  
  observeEvent(input$map_marker_click, {
    click <- input$map_marker_click
    shelter <- shelter_cleaned[which(shelter_cleaned$LAT == click$lat & shelter_cleaned$LON == click$lng), ]$LOCATION_NAME
    output$pred_model <- renderPlot({
      makeTimeSeriesPlot(shelter)
    })
  })
  
  observeEvent(input$map_shape_click, {
    click <- input$map_shape_click
    neighbourhood <- click$id
    if (neighbourhood %in% unique(neighbourhood_joined$Neighbourhood)) {
      output$pred_model <- renderPlot({
        makeTimeSeriesPlotN(neighbourhood)
      })
    }
  })
  
}

# Run app
shinyApp(ui, server)