## Baseline

We initiated our analysis by establishing a baseline model, serving as a benchmark for subsequent comparisons.

### PCA

> Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed. -- [What is PCA](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186#:~:text=Principal%20component%20analysis%2C%20or%20PCA,more%20easily%20visualized%20and%20analyzed.)

PCA allows us to extract insightful information from a large dataset. In this project, we used PCA to reduce the dimensionality of our dataset and then use the reduced dataset to train a prediction model. We explored several models including linear regression, SVM, and random forest. We decided to use Random Forest as the main benchmark.

#### Random Forest + Location Metadata + Sliding Window

We tried Random Forest using the neighbourhood features from the `location-metadata` folder in this repo, together with
a sliding window of the previous day's service user count. The Random Forest model achieved an RMSE of 15.12 and MAE of 6.76.

The source code resides in `random_forest_with_sliding_window.ipynb`.
