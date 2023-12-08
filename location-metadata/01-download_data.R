#### Preamble ####
# Purpose: Downloads 2021 Neighbourhood Profiles dataset from Open Data Toronto and saves to file 'neighbourhood-profiles-2021-158-model.csv'


#### Workspace setup ####
library(opendatatoronto)
library(dplyr)
library(sf)

#### Download Neighbourhood Profiles data ####
# get package
package <- show_package("6e19a90f-971c-46b3-852c-0c48c436d1fc")

# get all resources for this package
resources <- list_package_resources("6e19a90f-971c-46b3-852c-0c48c436d1fc")

# load the 2021 Neighbourhood Profiles dataset
data <- filter(resources, name == 'neighbourhood-profiles-2021-158-model') %>% get_resource()

# get census neighbourhood profile data and change to dataframe
neighbourhood_profiles <- do.call(rbind.data.frame, data["hd2021_census_profile"])

#### Download Neighbourhood GEOJSON data ####
# get package
package <- show_package("neighbourhoods")

# get all resources for this package
resources <- list_package_resources("neighbourhoods")

# load the Neighbourhoods geojson
neighbourhood_geojson <- filter(resources, name == 'Neighbourhoods - 4326.geojson') %>% get_resource()

#### Save data ####
write_csv(neighbourhood_profiles, file='neighbourhood-profiles-2021-158-model.csv')
st_write(neighbourhood_geojson, dsn = 'Neighbourhoods - 4326.geojson')
