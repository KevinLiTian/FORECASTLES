#### Preamble ####
# Purpose: Downloads 2021 Neighbourhood Profiles dataset from Open Data Toronto and saves to file 'neighbourhood-profiles-2021-158-model.csv'


#### Workspace setup ####
library(opendatatoronto)
library(dplyr)

#### Download data ####
# get package
package <- show_package("6e19a90f-971c-46b3-852c-0c48c436d1fc")

# get all resources for this package
resources <- list_package_resources("6e19a90f-971c-46b3-852c-0c48c436d1fc")

# load the 2021 Neighbourhood Profiles dataset
data <- filter(resources, name == 'neighbourhood-profiles-2021-158-model') %>% get_resource()

# get census neighbourhood profile data and change to dataframe
df <- do.call(rbind.data.frame, data["hd2021_census_profile"])

#### Save data ####
write_csv(df, file='neighbourhood-profiles-2021-158-model.csv')
