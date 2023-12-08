## Trained models
Consists of 2 types of deep learning models, MLP and transformer models. Both are trained on the pca location features along with location id, shelter id and past days counts. The transformer models perform better than the mlp models in general due to better learning of spare representations.
The testing data is the 2023 data.
Further to improve numerical stability, the models are trained to predict log user counts.

#### MLP
The mlp models are trained on 2 types of data, one where we provide the model with the type, day, month, sector and location id of a shelter; and other where we also provide the capacity of 2021 and 2022 shelter data. 

A sequential model consisting of fully connected layers, predicting the next day's user count for a given shelter location.

#### Transformers
Transformer models, one consisting of only the encoder, and other consisting of the full architecture. The models paprams are in the model name and the saved checkpoint.

##### Encoder only Transformer
Takes in sequence data of the past 30 days, creates a latent representation which is flattened and passed through fully connected laers for predictions.

## Overview of File
`data.py`: File consisting of functions and classes for loading data, cleaning and prepping for feeding to deep learning models.
 - `*Dataset`: Dataset wrappers that process the input pandas dataframes to tensors.
 - `*Dataloader`: Responsible for batching and deploying to models for evaluation.
 - `load_*_data`: Funtions for loading various raw datasets, processing, cleaning and normalizing into data frames for passing to Datasets.

`model.py`: File consisting of pytorch models and classes for time encodings
 - `MLP`: Simple Multi-Layer Perceptron with relu activatins and no window.
 - `TimeModel`: Encoder-only transformer model tiwh MLP head to predict counts. Has customisable window, trained on 30.  
 - `Full_Transformer`: Transformer Encoder-Decoder model, the encoder conditioned on features of window size, the Decoder takes as prompt the user values for past 30 days and a causal mask to predict the next day for any size of window.
 - `PositionalEncoding`: Temporal encoding mask for encoding time.

`train.py`: File containing the main training loop, evaluation and code to generate evaluation and loss plots.