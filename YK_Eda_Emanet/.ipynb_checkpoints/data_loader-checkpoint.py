import pandas as pd

class DataLoader:
    def __init__(self, data_path, delimiter=";"):
        self.data_path = data_path
        self.delimiter = delimiter

    def load_data(self):
        data = pd.read_csv(self.data_path, delimiter=self.delimiter)
        data.drop_duplicates(inplace=True)
        return data