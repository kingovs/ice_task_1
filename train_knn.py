# train_knn.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

class KNNTrainer:
    def __init__(self, data_path='face_data.csv', model_path='knn_model.pkl'):
        self.data_path = data_path
        self.model_path = model_path

    def train_knn(self):
        df = pd.read_csv(self.data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        
        joblib.dump(knn, self.model_path)
        print(f"KNN model trained and saved to {self.model_path}")

if __name__ == "__main__":
    trainer = KNNTrainer()
    trainer.train_knn()
