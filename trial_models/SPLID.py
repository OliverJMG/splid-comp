import os
import pandas as pd
from torch.utils.data import Dataset


class SPLID(Dataset):
    def __init__(self, label_file, root_dir):
        self.node_labels = pd.read_csv(label_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.no)