The mlp models are trained on 2 types of data, one where we provide the model with the type, day, month, sector and location id of a shelter; and other where we also provide the capacity of 2021 and 2022 shelter data. 

The testing data is the 2023 data.

Notice that the number of input features is very low, so it is difficult for the deep learning model to get good accuracy, but also the model performs better when we provide the capacity. It is better not only in terms of the predicted values being closer, but the distribution of prediction is also similar.

TODO: train model using location metadata instead of location ID as it will increase the number of features and it is possible to predict capacity using the metadata allowing us to remove the capacity. Removing capacity allows us to predict shelter demands in regions outside the data.