#### Preamble ####
# Purpose: Join neighbourhood profiles to shelters
# Author: Kimlin Chin
# Date: 2 November 2023


#### Workspace setup ####
library(tidyverse)

#### Prepare Data ####
# Get data
shelter_neighbourhoods <- read_csv("shelter_neighbourhoods.csv")
neighbourhood_features <- read_csv("neighbourhood_profiles.csv")
neighbourhood_features_pca <- read_csv("neighbourhood_profiles_pca.csv")

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

# fix typos
neighbourhood_features <- neighbourhood_features %>%
  mutate(`Neighbourhood Name` = str_replace(`Neighbourhood Name`, 'East End Danforth', 'East End-Danforth'))

neighbourhood_features_pca <- neighbourhood_features_pca %>%
  mutate(`Neighbourhood Name` = str_replace(`Neighbourhood Name`, 'East End Danforth', 'East End-Danforth'))

shelter_neighbourhoods <- shelter_neighbourhoods %>%
  mutate(`Neighbourhood` = str_replace(`Neighbourhood`, 'North St.James Town', 'North St. James Town'))

# drop columns with NA values
neighbourhood_features <- neighbourhood_features[,-c(180, 195)]

# combine data
shelter_occupancy <- rbind(shelter_occupancy_2021, shelter_occupancy_2022, shelter_occupancy_2023)

shelter_neighs <- shelter_neighbourhoods %>%
  select(LOCATION_ID, Neighbourhood, LAT, LON)

# all features, no pca
df2 <- left_join(x=shelter_neighs,
                 y=neighbourhood_features,
                 by = c("Neighbourhood" = "Neighbourhood Name")
                 )

df3 <- left_join(x=shelter_occupancy,
                 y=df2
                 )

# drop shelters that have no neighbourhood
df3 <- df3 %>%
  drop_na(Neighbourhood)

# fix 2 rows with missing values
rowname1 <- na.omit(rownames(df3)[df3$LOCATION_ID==1200 & df3$OCCUPANCY_DATE == as.Date("2021-12-30")])
df3[rowname1, "LOCATION_NAME"] <- "HFS Placer"
df3[rowname1, "PROGRAM_NAME"] <- "Homes First Society - Placer Mixed Adult Program"
df3[rowname1, "PROGRAM_MODEL"] <- "Emergency"
df3[rowname1, "OVERNIGHT_SERVICE_TYPE"] <- "Shelter"
df3[rowname1, "PROGRAM_AREA"] <- "Base Shelter and Overnight Services System"

rowname1 <- na.omit(rownames(df3)[df3$LOCATION_ID==1200 & df3$OCCUPANCY_DATE == as.Date("2021-12-31")])
df3[rowname1, "LOCATION_NAME"] <- "HFS Placer"
df3[rowname1, "PROGRAM_NAME"] <- "Homes First Society - Placer Mixed Adult Program"
df3[rowname1, "PROGRAM_MODEL"] <- "Emergency"
df3[rowname1, "OVERNIGHT_SERVICE_TYPE"] <- "Shelter"
df3[rowname1, "PROGRAM_AREA"] <- "Base Shelter and Overnight Services System"

write_csv(df3, file='shelter_neighbourhood_features.csv')


# with pca features
df2 <- left_join(x=shelter_neighs,
                 y=neighbourhood_features_pca,
                 by = c("Neighbourhood" = "Neighbourhood Name")
)

df3 <- left_join(x=shelter_occupancy,
                 y=df2
)

# drop shelters that have no neighbourhood
df3 <- df3 %>%
  drop_na(Neighbourhood)

# fix 2 rows with missing values
rowname1 <- na.omit(rownames(df3)[df3$LOCATION_ID==1200 & df3$OCCUPANCY_DATE == as.Date("2021-12-30")])
df3[rowname1, "LOCATION_NAME"] <- "HFS Placer"
df3[rowname1, "PROGRAM_NAME"] <- "Homes First Society - Placer Mixed Adult Program"
df3[rowname1, "PROGRAM_MODEL"] <- "Emergency"
df3[rowname1, "OVERNIGHT_SERVICE_TYPE"] <- "Shelter"
df3[rowname1, "PROGRAM_AREA"] <- "Base Shelter and Overnight Services System"

rowname1 <- na.omit(rownames(df3)[df3$LOCATION_ID==1200 & df3$OCCUPANCY_DATE == as.Date("2021-12-31")])
df3[rowname1, "LOCATION_NAME"] <- "HFS Placer"
df3[rowname1, "PROGRAM_NAME"] <- "Homes First Society - Placer Mixed Adult Program"
df3[rowname1, "PROGRAM_MODEL"] <- "Emergency"
df3[rowname1, "OVERNIGHT_SERVICE_TYPE"] <- "Shelter"
df3[rowname1, "PROGRAM_AREA"] <- "Base Shelter and Overnight Services System"


write_csv(df3, file='shelter_neighbourhood_features_pca.csv')

