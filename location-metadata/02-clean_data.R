#### Preamble ####
# Purpose: Clean data from file 'neighbourhood-profiles-2021-158-model.csv'


#### Workspace setup ####
library(tidyverse)

#### Clean Data ####
df <- read_csv("neighbourhood-profiles-2021-158-model.csv")

# remember the characteristics
characteristics <- df$`Neighbourhood Name`

# transpose data to have neighbourhoods as rows and characteristics as columns
df <- as.data.frame(t(df[,-1]))
colnames(df) <- characteristics
df <- df %>% rownames_to_column() %>% rename(`Neighbourhood Name` = rowname)

# drop language columns and incorrect value columns
df <- df[-c(383:385, 428:1446, 2255:2565)]

# fix dtypes of columns
df[, 4:ncol(df)] <- sapply(df[, 4:ncol(df)], as.numeric)

# remove sparse columns
sparse_colnames <- na.omit(colnames(df)[colSums(df==0) > 0.8*nrow(df)])
df <- df[ , !(names(df) %in% sparse_colnames)]

#### Save data ####
write_csv(df, file='neighbourhood_profiles.csv', na="")
