# Load libraries
library(shiny)
library(leaflet)
library(sf)
library(tidyverse)
library(lubridate)

# Get data
shelters <- read_csv("shelter.csv")
neighbourhoods <- read_csv("neighbourhood.csv")
neighbourhood_map <- st_read("Neighbourhoods - 4326.shp")

shelter_occupancy_2023 <- read_csv("daily_shelter_overnight_occupancy.csv")
shelter_occupancy_2022 <- read_csv("daily-shelter-overnight-service-occupancy-capacity-2022.csv")
shelter_occupancy_2021 <- read_csv("daily-shelter-overnight-service-occupancy-capacity-2021.csv")

# fix date format
shelter_occupancy_2021 <- shelter_occupancy_2021 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%y-%m-%d"))
shelter_occupancy_2022 <- shelter_occupancy_2022 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%y-%m-%d"))
shelter_occupancy_2023 <- shelter_occupancy_2023 %>%
  mutate(OCCUPANCY_DATE = as.Date(OCCUPANCY_DATE, format = "%Y-%m-%d"))
# combine data
shelter_occupancy <- rbind(shelter_occupancy_2021, shelter_occupancy_2022, shelter_occupancy_2023)

# Prepare data
shelter_data <- unique(shelter_occupancy %>% select(LOCATION_ID, LOCATION_NAME, LOCATION_ADDRESS, LOCATION_POSTAL_CODE, LOCATION_CITY))

neighbourhood_geometry <- neighbourhood_map %>%
  select(AREA_NA7, geometry) %>%
  rename(Neighbourhood = AREA_NA7)

neighbourhood_joined <- left_join(
  x=neighbourhoods,y=neighbourhood_geometry
  ) %>%
  mutate(date = make_date(year = year, month = month, day = day))

shelter_cleaned <- shelters %>%
  mutate(date = make_date(year = year, month = month, day = day))

individual_shelters <- unique(inner_join(shelter_data, shelter_cleaned) %>% select(LOCATION_ID, LOCATION_NAME, LAT, LON))

shelter_cleaned <- left_join(shelter_cleaned, shelter_data)

# prepare data for time plot
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

makeTimeSeriesPlot <- function (loc_name) {
  selected_shelter_data <- shelter_time_plot[shelter_time_plot$LOCATION_NAME == loc_name,]
  selected_shelter_data <- selected_shelter_data[!is.na(selected_shelter_data$LOCATION_NAME),]
  
  colors <- c("Capacity" = "blue", "Predicted Service User Count" = "purple", "Actual Service User Count" = "darkgreen")
  
  
  loc_cap <- shelter_occupancy[shelter_occupancy$LOCATION_NAME == loc_name,]
  loc_cap <- loc_cap[!is.na(loc_cap$LOCATION_NAME),]
  
  
  p <- ggplot() +
    geom_line(data = selected_shelter_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT, colour = "Actual Service User Count")) +
    geom_line(data = selected_shelter_data, mapping = aes(
      x = date, y = SERVICE_USER_COUNT_PRED, colour = "Predicted Service User Count")) +
    geom_line(data = loc_cap, mapping = aes(
      x = OCCUPANCY_DATE, y = CAPACITY_ACTUAL_BED, colour = "Capacity")) +
    geom_line(data = loc_cap, mapping = aes(
      x = OCCUPANCY_DATE, y = CAPACITY_ACTUAL_ROOM, colour = "Capacity")) +
    labs(title = str_c("Predicted service users for ", loc_name), x = "Date", y = "Service User Count", colour = "Legend") +
    scale_color_manual(values = colors)
  
  return (p)
}



# Create shiny app
ui <- bootstrapPage(
  tags$style(type = "text/css", "html, body {width:100%;height:100%}"),
  leafletOutput("map", width = "100%", height = "100%"),
  absolutePanel(top = 10, right = 10,
                plotOutput("pred_model", width = "20%", height = "20%")
  )
)

server <- function(input, output, session) {
  
  output$map <- renderLeaflet({
    # Use leaflet() here, and only include aspects of the map that
    # won't need to change dynamically (at least, not unless the
    # entire map is being torn down and recreated).
    leaflet() %>% addTiles() %>%
      addPolygons(
        data = neighbourhood_geometry,
        # set the color of the polygon
        color = "#E84A5F",
        popup = ~neighbourhood_geometry$Neighbourhood,
        # set the opacity of the outline
        opacity = 1,
        # set the stroke width in pixels
        weight = 1,
        # set the fill opacity
        fillOpacity = 0.6
      ) %>%
      addMarkers(
        data = individual_shelters,
        lng = ~LON,
        lat = ~LAT,
        popup = ~LOCATION_NAME,
        layerId = ~LOCATION_ID,
      )
  })

  observeEvent(input$map_click, { 
    event <- input$map_click
    clickAreaName <- individual_shelters$LOCATION_NAME[individual_shelters$LOCATION_ID == event$id][1]
    if (!is.na(clickAreaName)) {
        output$pred_model <- renderPlot({
        makeTimeSeriesPlot(clickAreaName)
      })
    } else {
      output$pred_model <- renderPlot({
        ggplot() + title(clickAreaName)
      })
    }
  })
}

shinyApp(ui, server)