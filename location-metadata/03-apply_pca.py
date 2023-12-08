# Preamble
# Purpose: Apply PCA to reduce number of columns in 'neighbourhood_profiles.csv'


# Import packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def prepare_data(dataframe):
    # Drop Neighbourhood Identifiers
    df = dataframe.drop(columns=["Neighbourhood Name", "Neighbourhood Number"])

    # Encode categorical variable
    le1 = LabelEncoder()
    df['TSNS 2020 Designation'] = le1.fit_transform(df['TSNS 2020 Designation'])

    # Drop columns with missing values
    df.dropna(axis='columns', inplace=True)

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data


def pca_to_dataset(original_data, x_pca):
    d = {
        'Neighbourhood Name': original_data['Neighbourhood Name'],
        'Neighbourhood Number': original_data['Neighbourhood Number']
    }
    for i in range(len(x_pca[0])):
        d['V' + str(i)] = x_pca[:, i]

    df = pd.DataFrame(d)
    return df


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('neighbourhood_profiles.csv')

    # Prepare the data
    X = prepare_data(data)

    # Apply PCA
    pca = PCA(n_components=0.99)  # keep 99% of variance
    X_pca = pca.fit_transform(X)
    print(f'Reduced number of features from {len(data.columns) - 2} to {len(X_pca[0])}!')

    # Create new dataframe with Neighbourhood Identifiers and PCA features
    dataset = pca_to_dataset(data, X_pca)

    # Save as csv
    dataset.to_csv('neighbourhood_profiles_pca.csv', index=False)
