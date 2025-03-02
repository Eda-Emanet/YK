from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

class ModelTrainer:
    def __init__(self, x_train, y_train, x_test, y_test, seed=42):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.seed = seed
        self.model = None
        self.y_pred = None

    def train_model(self):
        scale_pos = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        self.model = XGBClassifier(scale_pos_weight=scale_pos, random_state=self.seed)
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Test Seti Doğruluk Skoru:", accuracy)
        print("\nSınıflandırma Raporu:")
        print(classification_report(self.y_test, self.y_pred))
        f1 = f1_score(self.y_test, self.y_pred)
        return f1