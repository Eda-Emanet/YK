import pandas as pd

class DataLoader:
    def __init__(self, data_path, delimiter=";", seed=42):
        self.data_path = data_path
        self.delimiter = delimiter
        self.seed = seed

    def load_data(self):
        data = pd.read_csv(self.data_path, delimiter=self.delimiter)
        return data