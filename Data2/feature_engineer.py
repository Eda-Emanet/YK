import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import warnings
import pandas as pd
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", category=FutureWarning)

class FeatureEngineer:
    def __init__(self, data, seed=42):
        self.data = data.copy()  # Orijinal veriyi korumak için bir kopya al
        self.seed = seed
        self.median_age = None
        self.occupation_to_education = None
        self.ordinal_encoder = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

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

    def get_feature_engineered_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test