import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, model, median_age, occupation_to_education, ordinal_encoder, columns):
        self.model = model
        self.median_age = median_age
        self.occupation_to_education = occupation_to_education
        self.ordinal_encoder = ordinal_encoder
        self.columns = columns

    def preprocess_new_data(self, data):
        if 'age' in data.columns:
            data['age'] = data['age'].fillna(self.median_age)
        data['age-hours'] = data['age'] * data['hours-per-week']
        data = self.fill_missing_education(data, self.occupation_to_education)
        if 'education_level' in data.columns:
            data["education_level_encoded"] = self.ordinal_encoder.transform(data[["education_level"]])
        data['education_work'] = data["education_level_encoded"] * data['hours-per-week']
        missing_columns = [col for col in ["sex", 'marital-status', "race", "occupation", "workclass",
                                           'has_capital_gain', 'is_American', "not_paying", "occupation_flag",
                                           'hours_type', 'is_college_grad'] if col not in data.columns]
        for col in missing_columns:
            data[col] = 0
        data = pd.get_dummies(data, columns=["sex", 'marital-status', "race", "occupation", "workclass",
                                             'has_capital_gain', 'is_American', "not_paying", "occupation_flag",
                                             'hours_type', 'is_college_grad'], drop_first=True)
        if 'education_level' in data.columns:
            data = data.drop(columns=["education_level"])
        return data

    @staticmethod
    def fill_missing_education(df, occupation_to_education):
        if 'education_level' in df.columns:
            df["education_level"] = df.apply(
                lambda row: occupation_to_education.get(row["occupation"], "HighGrad") if pd.isna(row["education_level"]) else row["education_level"],
                axis=1
            )
        return df

    def predict(self, data):
        processed_data = self.preprocess_new_data(data)
        predictions = self.model.predict(processed_data[self.columns])
        data["income"] = predictions
        return data

    def save_predictions(self, data, file_path):
        data.to_excel(file_path, index=False)