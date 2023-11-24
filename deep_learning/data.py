import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

def load_sequence_data(data_path):
    all_shelter_data = pd.read_csv(data_path, low_memory=False)
    all_shelter_data['OCCUPANCY_DATE'] = pd.to_datetime(all_shelter_data['OCCUPANCY_DATE'])
    # toronto_data = all_shelter_data[all_shelter_data["LOCATION_CITY"] == "Toronto"]
    toronto_data = all_shelter_data
    toronto_data['MONTH'] = toronto_data['OCCUPANCY_DATE'].dt.month
    toronto_data['DAY'] = toronto_data['OCCUPANCY_DATE'].dt.day
    toronto_data['YEAR'] = toronto_data['OCCUPANCY_DATE'].dt.year
    # df[df["LOCATION_CITY"]=="Toronto"].groupby(["OCCUPANCY_DATE", "LOCATION_ID", "SHELTER_ID", "SECTOR", "PROGRAM_MODEL", "CAPACITY_TYPE"])
    drop_cols = ['_id', 'ORGANIZATION_ID', 'ORGANIZATION_NAME', 'SHELTER_GROUP',
                 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_CITY', 'LOCATION_PROVINCE', 'PROGRAM_ID',
                 'PROGRAM_NAME', 'OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA', 'CAPACITY_FUNDING_BED',
                 'OCCUPIED_BEDS', 'UNOCCUPIED_BEDS', 'CAPACITY_FUNDING_ROOM', 'OCCUPIED_ROOMS', 'UNOCCUPIED_ROOMS',
                 'OCCUPANCY_RATE_ROOMS', "OCCUPANCY_RATE_BEDS", 'UNAVAILABLE_BEDS','UNAVAILABLE_ROOMS',
                 "LOCATION_POSTAL_CODE", "Neighbourhood", "Neighbourhood Number", 'LAT', 'LON',
                 'CAPACITY_ACTUAL_BED', 'CAPACITY_ACTUAL_ROOM'
                 ]
    toronto_data_dr = toronto_data.drop(columns=drop_cols)
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

    final_df = toronto_data_dr_nan
    # occu_final_df = toronto_data_dr_nan
    # occu_final_df = occu_final_df[occu_final_df['YEAR']==2023]
    # info = {"test_dated": occu_final_df.reset_index(drop=True)}
    final_df = final_df[final_df["V1"].notna()]
    final_df = final_df.fillna(value=0.0)
    test = final_df[final_df['YEAR']==2023]
    train = final_df[(final_df['YEAR']==2021) | (final_df['YEAR']==2022)]
    train = train.drop(columns=['YEAR'])
    test = test.drop(columns=['YEAR'])
    X_train, X_test, y_train, y_test = train, test, train['SERVICE_USER_COUNT'], test['SERVICE_USER_COUNT']
    info = {"X_train": X_train.copy().reset_index(drop=True),
            "X_test":  X_test.copy().reset_index(drop=True),
            "y_train": y_train.copy().reset_index(drop=True),
            "y_test":  y_test.copy().reset_index(drop=True),
            "final_df": final_df.copy().reset_index(drop=True)}
    sc = StandardScaler()
    # print(list(X_train.select_dtypes(include=['object']).columns))
    # Fit_transform on train data
    cols = list(X_train.columns)
    cols.remove("OCCUPANCY_DATE")
    X_train[cols] = sc.fit_transform(X_train[cols])

    # Transform on test data
    X_test[cols] = sc.transform(X_test[cols])
    info["scaler"] = sc
    print("Data loaded")
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True), info

def load_data(include_capacity=True):
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
    drop_cols = ['_id', 'ORGANIZATION_ID', 'ORGANIZATION_NAME', 'SHELTER_ID', 'SHELTER_GROUP',
                 'LOCATION_ID', 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_CITY', 'LOCATION_PROVINCE', 'PROGRAM_ID',
                 'PROGRAM_NAME', 'OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA', 'CAPACITY_FUNDING_BED',
                 'OCCUPIED_BEDS', 'UNOCCUPIED_BEDS', 'CAPACITY_FUNDING_ROOM', 'OCCUPIED_ROOMS', 'UNOCCUPIED_ROOMS',
                 'OCCUPANCY_RATE_ROOMS', "OCCUPANCY_RATE_BEDS", 'UNAVAILABLE_BEDS','UNAVAILABLE_ROOMS'
                 ]
    if not include_capacity:
        drop_cols.extend(['CAPACITY_ACTUAL_BED', 'CAPACITY_ACTUAL_ROOM'])
    toronto_data_dr = toronto_data.drop(columns=drop_cols)
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
    final_df = toronto_data_dr_nan.drop(columns=["postal_code", "LOCATION_POSTAL_CODE", 'OCCUPANCY_DATE'])
    occu_final_df = toronto_data_dr_nan.drop(columns=["postal_code", "LOCATION_POSTAL_CODE"])
    occu_final_df = occu_final_df[occu_final_df['YEAR']==2023]
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

    return X_train_scaled, X_test_scaled, y_train, y_test, occu_final_df


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
        # self.x = torch.from_numpy(data_x)
        # self.y = torch.from_numpy(data_y)
        cols = [col for col in data_x.columns if col[0]!='V']
        data_cols = [col for col in data_x.columns]
        data_cols.remove("OCCUPANCY_DATE")
        self.data_cols = data_cols
        cols.remove("OCCUPANCY_DATE")
        cols.remove("SERVICE_USER_COUNT")
        cols.remove("MONTH")
        cols.remove("DAY")
        data_x["GNUM"] = data_x.groupby(cols).ngroup()
        data_x["LOG_CNT"] = data_y
        self.x = data_x
        self.cumsum = np.cumsum(np.clip((data_x.groupby("GNUM").count()["OCCUPANCY_DATE"].to_numpy() - window), a_min=0, a_max=None))
        self.length = self.cumsum[-1]
        self.window = window
        self.gnum_subsets = []
        max_gnum = self.x["GNUM"].max()
        for gnum in tqdm.tqdm(range(max_gnum)):
            subset = self.x[self.x["GNUM"]==gnum]
            self.gnum_subsets.append((subset[self.data_cols], np.expand_dims(subset["LOG_CNT"].to_numpy(), axis=-1)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        diff = self.cumsum-index
        gnum = np.argmax(diff>0)
        if gnum==0:
          g_idx = index
        else:
          g_idx = index-self.cumsum[gnum-1]
        # subset_x, subset_y  = self.gnum_subsets[gnum]
        return torch.from_numpy(self.gnum_subsets[gnum][0].to_numpy())[g_idx:g_idx+self.window].float(), torch.from_numpy(self.gnum_subsets[gnum][1])[g_idx+self.window].float()


class SlowSequenceDataset(Dataset):
    def __init__(self, data_x, data_y, window=30):
        # self.x = torch.from_numpy(data_x)
        # self.y = torch.from_numpy(data_y)
        cols = [col for col in data_x.columns if col[0]!='V']
        data_cols = [col for col in data_x.columns]
        data_cols.remove("OCCUPANCY_DATE")
        self.data_cols = data_cols
        cols.remove("OCCUPANCY_DATE")
        cols.remove("SERVICE_USER_COUNT")
        cols.remove("MONTH")
        cols.remove("DAY")
        if "UNSCALED_SC" in cols:
          cols.remove("UNSCALED_SC")
        data_x["GNUM"] = data_x.groupby(cols).ngroup()
        data_x["LOG_CNT"] = data_y
        data_x['index_col'] = list(data_x.index)
        self.x = data_x
        self.cumsum = np.cumsum(np.clip((data_x.groupby("GNUM").count()["OCCUPANCY_DATE"].to_numpy() - window), a_min=0, a_max=None))
        self.length = self.cumsum[-1]
        self.window = window
        self.gnum_subsets = []
        max_gnum = self.x["GNUM"].max()
        for gnum in tqdm.tqdm(range(max_gnum)):
            subset = self.x[self.x["GNUM"]==gnum]
            self.gnum_subsets.append(subset.copy())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        diff = self.cumsum-index
        gnum = np.argmax(diff>0)
        if gnum==0:
          g_idx = index
        else:
          g_idx = index-self.cumsum[gnum-1]
        # subset = self.x[self.x["GNUM"]==gnum]
        # print(subset["index_col"])
        # return torch.from_numpy(subset[self.data_cols].to_numpy())[g_idx:g_idx+self.window].float(), torch.from_numpy(np.expand_dims(subset["LOG_CNT"].to_numpy(), axis=-1))[g_idx+self.window].float(), subset["index_col"].to_numpy()[g_idx+self.window], subset.iloc[g_idx:g_idx+self.window]
        # subset_x, subset_y  = self.gnum_subsets[gnum]
        return torch.from_numpy(self.gnum_subsets[gnum][self.data_cols].to_numpy())[g_idx:g_idx+self.window].float(), torch.from_numpy(np.expand_dims(self.gnum_subsets[gnum]["LOG_CNT"].to_numpy(), axis=-1))[g_idx+self.window].float(), self.gnum_subsets[gnum]["index_col"].to_numpy()[g_idx+self.window], self.gnum_subsets[gnum].iloc[g_idx:g_idx+self.window]
    def get_gid(self, index):
        diff = self.cumsum-index
        gnum = np.argmax(diff>0)
        if gnum==0:
          g_idx = index
        else:
          g_idx = index-self.cumsum[gnum-1]
        return g_idx

    def get_df(self):
        return self.x

    def get_gnum(self, index):
        diff = self.cumsum-index
        gnum = np.argmax(diff>0)
        return gnum

    def get_df_index(self, index):
        gnum = self.get_gnum(index)
        gidx = self.get_gid(index)
        subset = self.x[self.x["GNUM"]==gnum]
        return subset["index_col"][gidx+self.window]


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

class FullTransSequenceDataset(Dataset):
    def __init__(self, data_x, data_y, window=30):
        # self.x = torch.from_numpy(data_x)
        # self.y = torch.from_numpy(data_y)
        cols = [col for col in data_x.columns if col[0]!='V']
        data_cols = [col for col in data_x.columns]
        data_cols.remove("OCCUPANCY_DATE")
        self.data_cols = data_cols
        cols.remove("OCCUPANCY_DATE")
        cols.remove("SERVICE_USER_COUNT")
        cols.remove("MONTH")
        cols.remove("DAY")
        data_x["GNUM"] = data_x.groupby(cols).ngroup()
        data_x["LOG_CNT"] = data_y
        self.x = data_x
        self.cumsum = np.cumsum(np.clip((data_x.groupby("GNUM").count()["OCCUPANCY_DATE"].to_numpy() - window), a_min=0, a_max=None))
        self.length = self.cumsum[-1]
        self.window = window
        self.gnum_subsets = []
        max_gnum = self.x["GNUM"].max()
        for gnum in tqdm.tqdm(range(max_gnum)):
            subset = self.x[self.x["GNUM"]==gnum]
            self.gnum_subsets.append((subset[self.data_cols], np.expand_dims(subset["LOG_CNT"].to_numpy(), axis=-1)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        diff = self.cumsum-index
        gnum = np.argmax(diff>0)
        if gnum==0:
          g_idx = index
        else:
          g_idx = index-self.cumsum[gnum-1]
        # subset_x, subset_y  = self.gnum_subsets[gnum]
        return torch.from_numpy(self.gnum_subsets[gnum][0].to_numpy())[g_idx:g_idx+self.window].float(), torch.from_numpy(self.gnum_subsets[gnum][1])[g_idx:g_idx+self.window].float(), torch.from_numpy(self.gnum_subsets[gnum][1])[g_idx+1:g_idx+self.window+1].float()
if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, info = load_sequence_data("./shelter_neighbourhood_features_pca.csv")
    df = info["final_df"]
    print(df[df.columns[:50]].info())
    # train_dataset = DefaultDataset(X_train_scaled.to_numpy(), np.asarray(y_train))
    # test_dataset = DefaultDataset(X_test_scaled.to_numpy(), np.asarray(y_test))
    # # train_dataset = SequenceDataset(X_train_scaled.to_numpy(), np.log10(np.expand_dims(np.asarray(y_train), axis=-1)))
    # # test_dataset = SequenceDataset(X_test_scaled.to_numpy(), np.log10(np.expand_dims(np.asarray(y_test), axis=-1)))
    #
    # x, y = train_dataset[10]
    # print(x.shape, y.shape)
    #
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)
    #
    # print(f"Training data batches {len(train_dataloader)}")
    # print(f"Testing data batches {len(test_dataloader)}")
