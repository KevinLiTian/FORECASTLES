import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_data():
    occupancy_data_2023 = pd.read_csv("../data/occupancy/Daily_shelter_overnight_occupancy.csv", low_memory=False)
    occupancy_data_2021 = pd.read_csv("../data/occupancy/daily-shelter-overnight-service-occupancy-capacity-2021.csv",
                                      low_memory=False)
    occupancy_data_2022 = pd.read_csv("../data/occupancy/daily-shelter-overnight-service-occupancy-capacity-2022.csv",
                                      low_memory=False)
    occupancy_data_2022['OCCUPANCY_DATE'] = pd.to_datetime(occupancy_data_2022['OCCUPANCY_DATE'],
                                                           format='%y-%m-%d').dt.strftime('%Y-%m-%d')
    occupancy_data_2021['OCCUPANCY_DATE'] = pd.to_datetime(occupancy_data_2021['OCCUPANCY_DATE'],
                                                           format='%y-%m-%d').dt.strftime('%Y-%m-%d')
    all_shelter_data = pd.concat([occupancy_data_2023, occupancy_data_2022, occupancy_data_2021])
    all_shelter_data['OCCUPANCY_DATE'] = pd.to_datetime(all_shelter_data['OCCUPANCY_DATE'], format='%Y-%m-%d')
    toronto_data = all_shelter_data[all_shelter_data["LOCATION_CITY"] == "Toronto"]
    toronto_data['MONTH'] = toronto_data['OCCUPANCY_DATE'].dt.month
    toronto_data['DAY'] = toronto_data['OCCUPANCY_DATE'].dt.day
    toronto_data['YEAR'] = toronto_data['OCCUPANCY_DATE'].dt.year
    toronto_data_dr = toronto_data.drop(
        columns=['_id', 'OCCUPANCY_DATE', 'ORGANIZATION_ID', 'ORGANIZATION_NAME', 'SHELTER_ID', 'SHELTER_GROUP',
                 'LOCATION_ID', 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_CITY', 'LOCATION_PROVINCE', 'PROGRAM_ID',
                 'PROGRAM_NAME', 'OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA', 'CAPACITY_FUNDING_BED',
                 'OCCUPIED_BEDS', 'UNOCCUPIED_BEDS', 'CAPACITY_FUNDING_ROOM', 'OCCUPIED_ROOMS', 'UNOCCUPIED_ROOMS',
                 'OCCUPANCY_RATE_ROOMS', "OCCUPANCY_RATE_BEDS"
                 ])
    toronto_data_dr_nan = toronto_data_dr[toronto_data_dr["PROGRAM_MODEL"].notna()]
    # Creating list of dummy columns
    to_get_dummies_for = ['SECTOR']

    # Creating dummy variables
    toronto_data_dr_nan = pd.get_dummies(data=toronto_data_dr_nan, columns=to_get_dummies_for)

    # Mapping overtime and attrition
    dict_prog_mod = {'Emergency': 1, 'Transitional': 0}
    dict_cap_type = {'Bed Based Capacity': 1, 'Room Based Capacity': 0}

    toronto_data_dr_nan['prog_mod'] = toronto_data_dr_nan["PROGRAM_MODEL"].map(dict_prog_mod)
    toronto_data_dr_nan['cap_type'] = toronto_data_dr_nan["CAPACITY_TYPE"].map(dict_cap_type)
    toronto_data_dr_nan = toronto_data_dr_nan.drop(columns=["PROGRAM_MODEL", "CAPACITY_TYPE"])
    toronto_data_dr_nan["postal_code"] = toronto_data_dr_nan['LOCATION_POSTAL_CODE'].apply(lambda x: x.split(" ")[0])
    toronto_data_dr_nan["postal_code"] = toronto_data_dr_nan["postal_code"].astype('category')
    toronto_data_dr_nan["postal_code_num"] = pd.Categorical(toronto_data_dr_nan["postal_code"]).codes
    final_df = toronto_data_dr_nan.drop(columns=["postal_code", "LOCATION_POSTAL_CODE"])
    final_df = final_df.fillna(value=0.0)
    test = final_df[final_df['YEAR']==2023]
    train = final_df[(final_df['YEAR']==2021) | (final_df['YEAR']==2022)]
    train = train.drop(columns=['YEAR'])
    test = test.drop(columns=['YEAR'])
    X_train, X_test, y_train, y_test = train.drop(columns=['SERVICE_USER_COUNT']), test.drop(columns=['SERVICE_USER_COUNT']), train['SERVICE_USER_COUNT'], test['SERVICE_USER_COUNT']
    sc = StandardScaler()

    # Fit_transform on train data
    X_train_scaled = sc.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Transform on test data
    X_test_scaled = sc.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


class DefaultDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.from_numpy(data_x)
        self.y = torch.from_numpy(data_y)
        self.length = len(data_y)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()


class SequenceDataset(Dataset):
    def __init__(self, data_x, data_y, window=30):
        self.x = torch.from_numpy(data_x)
        self.y = torch.from_numpy(data_y)
        self.length = len(data_y)-window+1
        self.window = window

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index:index+self.window].float(), self.y[index+self.window-1].float()


class NewsDataset(Dataset):
    def __init__(self, data_x, data_y, news_data, window=30):
        self.x = torch.from_numpy(data_x)
        self.y = torch.from_numpy(data_y)
        self.length = len(data_y)-window+1
        self.window = window
        self.news_data = news_data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.news_data[index:index+self.window], self.x[index:index+self.window].float(), self.y[index+self.window-1].float()


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()
    train_dataset = DefaultDataset(X_train_scaled.to_numpy(), np.asarray(y_train))
    test_dataset = DefaultDataset(X_test_scaled.to_numpy(), np.asarray(y_test))
    # train_dataset = SequenceDataset(X_train_scaled.to_numpy(), np.log10(np.expand_dims(np.asarray(y_train), axis=-1)))
    # test_dataset = SequenceDataset(X_test_scaled.to_numpy(), np.log10(np.expand_dims(np.asarray(y_test), axis=-1)))

    x, y = train_dataset[10]
    print(x.shape, y.shape)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)

    print(f"Training data batches {len(train_dataloader)}")
    print(f"Testing data batches {len(test_dataloader)}")
