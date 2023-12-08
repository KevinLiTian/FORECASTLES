#### Preamble ####
# Purpose: Add longitude latitude coordinates to shelter data
# Author: Kimlin Chin
# Date: 2 November 2023


#### Workspace setup ####
library(tmap)
library(tmaptools)
library(tidyverse)

#### Prepare Data ####
# get data
shelter_occupancy_2023 <- read_csv("daily_shelter_overnight_occupancy.csv")
shelter_occupancy_2022 <- read_csv("daily-shelter-overnight-service-occupancy-capacity-2022.csv")
shelter_occupancy_2021 <- read_csv("daily-shelter-overnight-service-occupancy-capacity-2021.csv")

# combine data
shelter_occupancy <- rbind(shelter_occupancy_2021, shelter_occupancy_2022, shelter_occupancy_2023)

## Add Shelter geocoding ##

# take subset with just location addresses
shelter_locations <- shelter_occupancy %>%
  select(
    LOCATION_ID,
    LOCATION_ADDRESS,
    LOCATION_POSTAL_CODE,
    LOCATION_CITY,
    LOCATION_PROVINCE
    )

shelter_locations <- unique(shelter_locations)

# fix one shelter missing city and province
rowname1 <- na.omit(rownames(shelter_locations)[shelter_locations$LOCATION_ID==1440])
shelter_locations[rowname1, "LOCATION_CITY"] <- "Toronto"
shelter_locations[rowname1, "LOCATION_PROVINCE"] <- "ON"

# remove missing values
shelter_locations <- na.omit(shelter_locations)

# fix wrong city value
rowname1 <- na.omit(rownames(shelter_locations)[shelter_locations$LOCATION_ID==1128])
shelter_locations[rowname1, "LOCATION_CITY"] <- "Toronto"

# only keep shelters in the City of Toronto
shelter_locations <- shelter_locations %>%
  filter(startsWith(LOCATION_POSTAL_CODE, 'M'))

# create new column with full address
shelter_locations <- shelter_locations %>%
  mutate(
    LOCATION_FULL_ADDRESS = str_c(
        str_trim(LOCATION_ADDRESS), ', ',
        str_trim(LOCATION_CITY), ', ',
        str_trim(LOCATION_PROVINCE)
      )
    )

# Add geographical coordinates
shelter_locations <- shelter_locations %>%
  mutate(COORDS = geocode_OSM(LOCATION_FULL_ADDRESS))

shelter_latlon <- shelter_locations %>%
  mutate(
    LAT = COORDS[,'lat'],
    LON = COORDS[,'lon']
    )

shelter_latlon <- select(shelter_latlon, -COORDS)

#### Save data ####
write_csv(shelter_latlon, file='shelter_locations.csv')
