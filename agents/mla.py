import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MachineLearningAgent:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "MLP": MLPClassifier(max_iter=300, random_state=42),
            "SVM": SVC(probability=True, kernel='rbf'),
            "NaiveBayes": GaussianNB(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None

    def train_models(self, df):
        """Original method - expects 'Label' column"""
        if 'Label' not in df.columns:
            raise ValueError("DataFrame must contain a 'Label' column")

        X = df.drop(columns=["Label"])
        y = df["Label"]
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_accuracy = 0
        print("[MLA] Training models...")
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"[MLA] {name}: {acc:.3f} accuracy")
                if acc > best_accuracy:
                    best_accuracy = acc
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:
                print(f"[MLA] Training error for {name}: {e}")

        print(f"[MLA] Best model: {self.best_model_name} ({best_accuracy:.3f})")

    def train_model(self, files_list):
        """Method for compatibility with model.py - expects list of dictionaries"""
        print("[MLA] Preparing training data...")

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(files_list)

        # Identify feature and target columns
        feature_cols = [col for col in df.columns if col.startswith('feature')]
        target_col = 'true_label'

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df[feature_cols]
        y = df[target_col]
        self.feature_columns = feature_cols

        if len(X) < 10:
            print("[MLA] Warning: Very small dataset for training")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_accuracy = 0
        print("[MLA] Training models...")

        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"[MLA] {name}: {acc:.3f} accuracy")

                if acc > best_accuracy:
                    best_accuracy = acc
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:
                print(f"[MLA] Training error for {name}: {e}")

        print(f"[MLA] Best model selected: {self.best_model_name} (accuracy: {best_accuracy:.3f})")

    def predict(self, file_dict):
        """Predict for a single file (dictionary format)"""
        if self.best_model is None or self.feature_columns is None:
            raise ValueError("Model has not been trained yet")

        # Extract features using the same order as during training
        features = []
        for col in self.feature_columns:
            features.append(file_dict.get(col, 0.0))

        # Use a DataFrame instead of a numpy array to preserve column names
        X = pd.DataFrame([features], columns=self.feature_columns)

        try:
            prediction = self.best_model.predict(X)[0]
            probability = self.best_model.predict_proba(X)[0]
            confidence_score = max(probability)
            return prediction, confidence_score, self.best_model_name
        except Exception as e:
            print(f"[MLA] Prediction error: {e}")
            return 0, 0.5, "default"

    def predict_batch(self, df_sim):
        """Predict for multiple files (DataFrame format)"""
        if self.best_model is None or self.feature_columns is None:
            raise ValueError("Model has not been trained yet")

        X_sim = df_sim[self.feature_columns]
        predictions = self.best_model.predict(X_sim)
        probabilities = self.best_model.predict_proba(X_sim)[:, 1]  # Probability of class 1 (malicious)

        df_result = df_sim.copy()
        df_result["Prediction"] = predictions
        df_result["Score_ML"] = probabilities
        df_result["Model_Used"] = self.best_model_name
        return df_result

    def entrainer_modeles(self, df_train):
        """French alias - compatibility with PlanningAgent"""
        return self.train_models(df_train)

    def predire(self, df_sim):
        """French alias - compatibility with PlanningAgent"""
        return self.predict_batch(df_sim)
