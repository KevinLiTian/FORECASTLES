## Install the libraries
install.packages('tidyverse')
install.packages('readr')
install.packages('sf')
install.packages('ggplot2')
install.packages('ggthemes')

## Import the packages
library(tidyverse)
library(readr)
library(sf)
library(ggplot2)
library(ggthemes)

## OPTIONAL : set your working directory using 
# setwd(<location>)


## Import shapefile for Toronto map. (may need to download files linked below)

# Idea: https://stackoverflow.com/questions/76854004/r-making-a-map-of-canadian-postal-codes

# A forward sortation area (FSA) is a way to designate a geographical unit based on the first three characters in a Canadian postal code. All postal codes that start with the same three characters—for example, K1A—are together considered an FSA.

# Map shape file of boundaries comes from Census Forward Sortation Area Boundary File from Statistics Canada: https://www150.statcan.gc.ca/n1/en/catalogue/92-179-X

# Reference guide: https://www150.statcan.gc.ca/n1/pub/92-179-g/92-179-g2021001-eng.htm

# Specific file from 2021 Census Year: https://www12.statcan.gc.ca/census-recensement/alternative_alternatif.cfm?l=eng&dispext=zip&teng=lfsa000b21a_e.zip&k=%20%20%20158240&loc=//www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lfsa000b21a_e.zip

# Feature columns: https://www150.statcan.gc.ca/n1/pub/92-179-g/2021001/tbl/tbl4_1-eng.htm

canada_map <- st_read("Toronto Postal Code shapefile\\lfsa000b21a_e\\lfsa000b21a_e.shp")


## Import data sets
##### Change the file locations to your own file locations

shelter_occupancy <- read_csv("C:/Users/miche/OneDrive - University of Toronto/UNIVERSITY COURSES/YEAR 4/EXTERNAL/Borealis AI project/daily-shelter-overnight-service-occupancy-capacity-2023.csv")


## add relevant features to data

# adds a column for general occupancy and capacity in a shelter
shelter_occupancy <- shelter_occupancy %>%
  mutate(CFSAUID = substr(LOCATION_POSTAL_CODE, 0, 3), 
         OCCUPIED_BEDS_OR_ROOMS = ifelse(is.na(OCCUPIED_BEDS), 0, OCCUPIED_BEDS) + 
           ifelse(is.na(OCCUPIED_ROOMS), 0, OCCUPIED_ROOMS), 
         CAPACITY_BEDS_OR_ROOMS = ifelse(is.na(CAPACITY_ACTUAL_BED), 0, CAPACITY_ACTUAL_BED) + 
           ifelse(is.na(CAPACITY_ACTUAL_ROOM), 0, CAPACITY_ACTUAL_ROOM))

# 
daily_shelter_occupancy_by_cfsauid <- select(shelter_occupancy, "CFSAUID", "OCCUPIED_BEDS_OR_ROOMS", "CAPACITY_BEDS_OR_ROOMS", "OCCUPANCY_DATE", "SHELTER_ID", "LOCATION_ID") %>%
  filter(startsWith(CFSAUID, 'M')) %>%
  group_by(CFSAUID, OCCUPANCY_DATE) %>%
  mutate(NUM_SHELTER_LOC = n_distinct(LOCATION_ID)) %>%
#  mutate(OCCUPANCY_DATE = str_c("20", OCCUPANCY_DATE)) %>%
  summarise(OCCUPIED_BEDS_OR_ROOMS = sum(OCCUPIED_BEDS_OR_ROOMS), CAPACITY_BEDS_OR_ROOMS = sum(CAPACITY_BEDS_OR_ROOMS), NUM_SHELTER_LOC = mean(NUM_SHELTER_LOC)) %>%
  mutate(OCCUPANCY_RATE =  round(OCCUPIED_BEDS_OR_ROOMS * 100 / CAPACITY_BEDS_OR_ROOMS, 2))


## Occupancy Map
ontario <-  canada_map[canada_map$PRUID == '35',]
toronto <- ontario[startsWith(ontario$CFSAUID, 'M'),]

occupancy_map <- full_join(toronto, daily_shelter_occupancy_by_cfsauid, by = "CFSAUID") %>%
  mutate(OCCUPANCY_DATE = replace_na(OCCUPANCY_DATE, ""))


## Generate maps and save map images to folder

occupancy_dates <- unique(occupancy_map$OCCUPANCY_DATE)

for (o_date in occupancy_dates) {
  df <- occupancy_map[occupancy_map$OCCUPANCY_DATE %in% c(o_date, ""),] %>%
    mutate(OCCUPANCY_DATE = o_date)
  
  graph1 <- ggplot(data = df) + 
    geom_sf(aes(fill = OCCUPANCY_RATE)) +
    # geom_sf_text(aes(label=str_c(CFSAUID, "\n", NUM_SHELTER_LOC)),
    #              fun.geometry = sf::st_centroid,
    #              colour='black') + 
    geom_sf_text(aes(label=str_c(NUM_SHELTER_LOC)),
                 fun.geometry = sf::st_centroid,
                 colour='black') + 
    theme_map() +
    scale_fill_continuous(name= "% occupancy", type = "viridis", na.value = "white", limits = c(0, 100)) + labs(title = str_c("Occupancy of Homeless Shelters in Toronto on ", o_date)) + theme(
      # plot.background = element_rect(fill = 'white'),
      legend.position = "right",
      legend.key.size = unit(32, 'mm'),
      legend.text = element_text(size = 15),
      legend.title = element_text(size = 20),
      legend.box.background = element_rect(fill = NULL),
      plot.title = element_text(size = 25)
    ) 
  
  
  # saves the images
  ggsave(
    filename = file.path("images", str_c(o_date, ".jpg")), 
    plot = graph1, 
    height = 300, width = 400,
    units = "mm", dpi = 100
  )
}