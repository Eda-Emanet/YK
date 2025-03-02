import pandas as pd
import numpy as np
import warnings
import pandas as pd
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", category=FutureWarning)

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()  # Orijinal veriyi korumak için bir kopya al
        self.val = None

    def preprocess(self):
        self.data.drop_duplicates(inplace=True)
        self.val = self.data[self.data['income'].isna()]
        self.data = self.data[self.data['income'].notnull()].copy()
        self.data.loc[:, "education-num"] = self.data.groupby("education_level")["education-num"].transform(lambda x: x.fillna(x.max()))
        self.data.loc[:, "income"] = self.data["income"].str.strip().map({'>50K': 1, '<=50K': 0})
        self.data.loc[:, self.data.select_dtypes(include=['object']).columns] = self.data.select_dtypes(include=['object']).apply(lambda x: x.str.strip())
        self.data.loc[:, 'education_level'] = self.data['education_level'].replace({
            'Preschool': 'dropout', '10th': 'dropout', '11th': 'dropout', '12th': 'dropout',
            '1st-4th': 'dropout', '5th-6th': 'dropout', '7th-8th': 'dropout', '9th': 'dropout',
            'HS-Grad': 'HighGrad', 'HS-grad': 'HighGrad',
            'Some-college': 'CommunityCollege', 'Assoc-acdm': 'CommunityCollege', 'Assoc-voc': 'CommunityCollege',
            'Bachelors': 'Bachelors', 'Masters': 'Masters', 'Prof-school': 'Masters', 'Doctorate': 'Doctorate'
        })
        self.data.loc[:, 'marital-status'] = self.data['marital-status'].replace({
            'Never-married': 'NotMarried', 'Married-AF-spouse': 'Married', 'Married-civ-spouse': 'Married',
            'Married-spouse-absent': 'NotMarried', 'Separated': 'Separated', 'Divorced': 'Separated',
            'Widowed': 'Widowed'
        })
        self.data.loc[:, 'occupation_flag'] = np.where(
            self.data['occupation'].isin(['Exec-managerial', 'Prof-specialty', 'Craft-repair']),
            1, 0
        )
        self.data.loc[:, 'net_capital'] = self.data['capital-gain'] - self.data['capital-loss']
        self.data.loc[:, 'has_capital_gain'] = self.data['capital-gain'].apply(lambda x: 1 if x > 0 else 0)
        self.data.loc[:, 'is_college_grad'] = self.data['education-num'].apply(lambda x: 1 if x >= 10 else 0)
        self.data.loc[:, 'hours_type'] = self.data['hours-per-week'].apply(self.map_hours_to_type)
        self.data.loc[:, 'is_American'] = self.data['native-country'].map(lambda x: 'United-States' if x == 'United-States' else 'Other')
        self.data.loc[:, "not_paying"] = np.where(self.data["workclass"] == "Without-pay", 1, 0)

    @staticmethod
    def map_hours_to_type(hours):
        if hours <= 20:
            return "Part-time"
        elif hours <= 40:
            return "Full-time"
        else:
            return "Overwork"

    def get_processed_data(self):
        return self.data