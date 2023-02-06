from torch.utils.data import Dataset
import pandas as pd
import torch

class DummyDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.label_columns = [x for x in self.df.columns if 'Label' in x]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx]
        y = torch.tensor(sample[self.label_columns].values)
        X = torch.tensor(sample.drop(columns = self.label_columns))

        return X, y


 
