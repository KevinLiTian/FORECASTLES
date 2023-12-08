## Adding Location Metadata

We enriched the initial shelter dataset by adding neighbourhood profile information from the 2021 Census. The files in this folder 
are numbered and should be run in ascending order to download, clean, prepare and join the neighbourhood data to the 
shelter datasets.

Since the 2021 Census dataset had over 2600 neighbourhood features, we applied Principal Component Analysis (PCA) to 
reduce this to just 119 features.

Two (2) CSV final output files are produced called `shelter_neighbourhood_features_pca.csv` and `shelter_neighbourhood_features.csv`
which contain the joined shelter and neighbourhood datasets with and without PCA respectively.