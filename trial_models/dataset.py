import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# Pytorch class for SPLID dataset
class SPLID(Dataset):
    def __init__(self, data_dir, label_file):

        self.train_dir = data_dir

        datalist = []

        # Searching for training data within the dataset folder
        for file in os.listdir(self.train_dir):
            if file.endswith(".csv"):
                datalist.append(os.path.join(self.train_dir, file))

        # Sort the training data and labels
        self.datalist = sorted(datalist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

        self.node_labels = pd.read_csv(label_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data_df = pd.read_csv(self.datalist[idx])
        data_df['TimeIndex'] = range(len(data_df))

        labels = self.node_labels[self.node_labels["ObjectID"] == idx+1].copy()

        labels_EW = labels[labels["Direction"] == "EW"].copy()
        labels_EW.insert(1, "EW", labels_EW["Node"] + '-' + labels_EW["Type"], True)
        labels_EW.drop(["Node", "Type", "Direction", "ObjectID"], axis=1, inplace=True)

        labels_NS = labels[labels["Direction"] == "NS"].copy()
        labels_NS.insert(1, "NS", labels_NS["Node"] + '-' + labels_NS["Type"], True)
        labels_NS.drop(["Node", "Type", "Direction", "ObjectID"], axis=1, inplace=True)

        # Merge the input data with the ground truth
        merged_df = pd.merge(data_df,
                             labels_EW.sort_values('TimeIndex'),
                             on=['TimeIndex'],
                             how='left')
        merged_df = pd.merge_ordered(merged_df,
                                     labels_NS.sort_values('TimeIndex'),
                                     on=['TimeIndex'],
                                     how='left')
        # Replace NaN values
        merged_df['EW'] = merged_df['EW'].fillna("none")
        merged_df['NS'] = merged_df['NS'].fillna("none")

        merged_df.drop("Timestamp", axis=1, inplace=True)

        return merged_df.drop(["EW", "NS"], axis=1).to_numpy(), merged_df[["EW", "NS"]].to_numpy()

