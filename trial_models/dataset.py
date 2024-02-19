import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class SPLID(Dataset):
    def __init__(self, datalist, labels, columns, classes=None):
        self.datalist = datalist
        self.labels = pd.read_csv(labels)
        self.columns = columns

        self.labels['merged_label'] = self.labels['Node'] + '-' + self.labels['Type']

        # Convert categorical data to numerical data
        self.le_type = LabelEncoder()
        if classes is None:
            self.le_type = LabelEncoder()
            self.labels['type_encoded'] = self.le_type.fit_transform(self.labels['merged_label'])
        else:
            self.le_type = LabelEncoder()
            self.le_type.fit(classes)
            self.labels['type_encoded'] = self.le_type.transform(self.labels['merged_label'])

        n_files = len(datalist)
        print('Loading {} files...'.format(n_files))

        frames = []

        for idx, file in enumerate(datalist):
            oid = int(Path(file).stem)
            data = pd.read_csv(file)
            data = data[columns].copy()

            labels = self.labels[(self.labels['ObjectID'] == oid)]

            for row in labels.itertuples():
                if row.Direction == 'NS':
                    data.loc[row.TimeIndex:, 'NS'] = row.type_encoded
                elif row.Direction == 'EW':
                    data.loc[row.TimeIndex:, 'EW'] = row.type_encoded
                else:  # direction is ES - 'end of sample'
                    data.loc[row.TimeIndex:, 'EW'] = row.type_encoded
                    data.loc[row.TimeIndex:, 'NS'] = row.type_encoded

            # Pad with 0s to 2208 rows before appending to ensure dimension uniformity
            data = data.reindex(range(2208), fill_value=0)
            data['ObjectID'] = oid
            frames.append(data)

            if idx % 50 == 0:
                print('Loaded file {} of {}'.format(idx, n_files))

        print('Joining dataframes...')

        df = pd.concat(frames, ignore_index=True)
        print('Done!')

        self.col_transformer = ColumnTransformer([("scaler", StandardScaler(), self.columns)], remainder="passthrough")
        self.col_transformer.fit(df)

        self.df = df
        self.ids = self.df['ObjectID'].unique()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data = self.df[self.df['ObjectID'] == self.ids[idx]]
        series = self.col_transformer.transform(data)[:, :len(self.columns)]
        labels = data[['EW', 'NS']].values

        series = torch.tensor(series, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return torch.permute(series, (1,0)), torch.permute(labels, (1,0))

    def get_id(self, idx):
        return self.ids[idx]

    def get_labels(self, idx):
        return self.labels[self.labels['ObjectID'] == self.get_id(idx)]

