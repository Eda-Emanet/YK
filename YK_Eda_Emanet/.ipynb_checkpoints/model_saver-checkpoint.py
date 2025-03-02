import joblib

class ModelSaver:
    def __init__(self, model, model_path):
        self.model = model
        self.model_path = model_path

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)
        return self.model