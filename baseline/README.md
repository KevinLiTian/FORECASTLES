# FORECASTLES

## Baseline

### PCA

> Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed. -- [What is PCA](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186#:~:text=Principal%20component%20analysis%2C%20or%20PCA,more%20easily%20visualized%20and%20analyzed.)

PCA allows us to extract insightful information from a large dataset. In this project, we used PCA to reduce the dimensionality of our dataset and then use the reduced dataset to train a prediction model. We tried several models including linear regression, SVM, and random forest. The result shows that PCA + random forest has the best performance.

The source code resides in `baseline/pca.ipynb`.

#### PCA + Random Forest

Since PCA + random forest has the best performance, we further tuned the hyperparameters of the random forest model. The result shows that the best hyperparameter is `n_estimators=400`.

The source code resides in `baseline/pca_random_forest.ipynb`.

#### PCA + Quantile Forest

We also tried quantile forest, which is a variant of random forest. The result shows that PCA + quantile forest performs worse than PCA + random forest.

The source code resides in `baseline/quantile_forest.ipynb`.
