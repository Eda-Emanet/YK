import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

class DataProcessor:
    def __init__(self, data, seed=42):
        self.data = data.copy()
        self.seed = seed
        self.median_age = None
        self.occupation_to_education = None
        self.ordinal_encoder = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.val = None

    def preprocess(self):
        self.data["education-num"] = self.data.groupby("education_level")["education-num"].transform(lambda x: x.fillna(x.max()))
        self.data["income"] = self.data["income"].str.strip().map({'>50K': 1, '<=50K': 0})
        self.data = self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.data['education_level'] = self.data['education_level'].replace({
            'Preschool': 'dropout', '10th': 'dropout', '11th': 'dropout', '12th': 'dropout',
            '1st-4th': 'dropout', '5th-6th': 'dropout', '7th-8th': 'dropout', '9th': 'dropout',
            'HS-Grad': 'HighGrad', 'HS-grad': 'HighGrad',
            'Some-college': 'CommunityCollege', 'Assoc-acdm': 'CommunityCollege', 'Assoc-voc': 'CommunityCollege',
            'Bachelors': 'Bachelors', 'Masters': 'Masters', 'Prof-school': 'Masters', 'Doctorate': 'Doctorate'
        })
        self.data['marital-status'] = self.data['marital-status'].replace({
            'Never-married': 'NotMarried', 'Married-AF-spouse': 'Married', 'Married-civ-spouse': 'Married',
            'Married-spouse-absent': 'NotMarried', 'Separated': 'Separated', 'Divorced': 'Separated',
            'Widowed': 'Widowed'
        })
        self.data['occupation_flag'] = np.where(
            self.data['occupation'].isin(['Exec-managerial', 'Prof-specialty', 'Craft-repair']),
            1, 0
        )
        self.data['net_capital'] = self.data['capital-gain'] - self.data['capital-loss']
        self.data['has_capital_gain'] = self.data['capital-gain'].apply(lambda x: 1 if x > 0 else 0)
        self.data['is_college_grad'] = self.data['education-num'].apply(lambda x: 1 if x >= 10 else 0)
        self.data['hours_type'] = self.data['hours-per-week'].apply(self.map_hours_to_type)
        self.data['is_American'] = self.data['native-country'].map(lambda x: 'United-States' if x == 'United-States' else 'Other')
        self.data["not_paying"] = np.where(self.data["workclass"] == "Without-pay", 1, 0)
        self.val = self.data[self.data['income'].isna()]
        self.data = self.data[self.data['income'].notnull()]

    @staticmethod
    def map_hours_to_type(hours):
        if hours <= 20:
            return "Part-time"
        elif hours <= 40:
            return "Full-time"
        else:
            return "Overwork"

    def split_data(self):
        x = self.data[['age', 'workclass', 'education_level', 'marital-status', 'occupation', 'race', 'sex',
                       'capital-loss', 'hours-per-week', 'net_capital', 'has_capital_gain', 'is_American',
                       "not_paying", "occupation_flag", 'hours_type', 'is_college_grad']]
        y = self.data[['income']]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed, stratify=y, shuffle=True)
        self.median_age = self.x_train['age'].median()
        self.x_train.loc[:, 'age'] = self.x_train['age'].fillna(self.median_age)
        self.x_train.loc[:, 'age-hours'] = self.x_train['age'] * self.x_train['hours-per-week']
        self.occupation_to_education = self.get_education_mode_mapping(self.x_train)
        self.x_train = self.fill_missing_education(self.x_train, self.occupation_to_education)
        self.ordinal_encoder = OrdinalEncoder(categories=[["dropout", "HighGrad", "CommunityCollege", "Bachelors", "Masters", "Doctorate"]],
                                              handle_unknown="use_encoded_value", unknown_value=-1)
        self.x_train.loc[:, "education_level_encoded"] = self.ordinal_encoder.fit_transform(self.x_train[["education_level"]])
        self.x_train.loc[:, 'education_work'] = self.x_train["education_level_encoded"] * self.x_train['hours-per-week']
        self.x_train = pd.get_dummies(self.x_train, columns=["sex", 'marital-status', "race", "occupation", "workclass",
                                                             'has_capital_gain', 'is_American', "not_paying", "occupation_flag",
                                                             'hours_type', 'is_college_grad'], drop_first=True)
        self.x_train = self.x_train.drop(columns=["education_level"])

    @staticmethod
    def get_education_mode_mapping(df):
        return df.groupby("occupation")["education_level"].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()

    @staticmethod
    def fill_missing_education(df, occupation_to_education):
        df.loc[:, "education_level"] = df.apply(
            lambda row: occupation_to_education.get(row["occupation"], "HighGrad") if pd.isna(row["education_level"]) else row["education_level"],
            axis=1
        )
        return df

    def preprocess_test_data(self):
        self.x_test.loc[:, 'age'] = self.x_test['age'].fillna(self.median_age)
        self.x_test.loc[:, 'age-hours'] = self.x_test['age'] * self.x_test['hours-per-week']
        self.x_test = self.fill_missing_education(self.x_test, self.occupation_to_education)
        self.x_test.loc[:, "education_level_encoded"] = self.ordinal_encoder.transform(self.x_test[["education_level"]])
        self.x_test.loc[:, 'education_work'] = self.x_test["education_level_encoded"] * self.x_test['hours-per-week']
        self.x_test = pd.get_dummies(self.x_test, columns=["sex", 'marital-status', "race", "occupation", "workclass",
                                                           'has_capital_gain', 'is_American', "not_paying", "occupation_flag",
                                                           'hours_type', 'is_college_grad'], drop_first=True)
        self.x_test = self.x_test.drop(columns=["education_level"])

    def preprocess_validation_data(self):
        self.val.loc[:, 'age'] = self.val['age'].fillna(self.median_age)
        self.val.loc[:, 'age-hours'] = self.val['age'] * self.val['hours-per-week']
        self.val = self.fill_missing_education(self.val, self.occupation_to_education)
        self.val.loc[:, "education_level_encoded"] = self.ordinal_encoder.transform(self.val[["education_level"]])
        self.val.loc[:, 'education_work'] = self.val["education_level_encoded"] * self.val['hours-per-week']
        self.val = pd.get_dummies(self.val, columns=["sex", 'marital-status', "race", "occupation", "workclass",
                                                     'has_capital_gain', 'is_American', "not_paying", "occupation_flag",
                                                     'hours_type', 'is_college_grad'], drop_first=True)
        self.val = self.val.drop(columns=["education_level"])
        return self.val

    def get_feature_engineered_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test