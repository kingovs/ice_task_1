# prepare_data.py
import cv2
import numpy as np
import os
import pandas as pd

class DataPreparer:
    def __init__(self, data_dir='data', save_path='face_data.csv'):
        self.data_dir = data_dir
        self.save_path = save_path
        self.face_data = []
        self.labels = []

    def prepare_data(self):
        names = os.listdir(self.data_dir)
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for name in names:
            dir_path = os.path.join(self.data_dir, name)
            for image_name in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image_name)
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(gray, 1.5, 5)
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100)).flatten()
                    self.face_data.append(face_img)
                    self.labels.append(name)
        
        self.save_to_csv()

    def save_to_csv(self):
        face_data = np.array(self.face_data)
        labels = np.array(self.labels).reshape(-1, 1)
        
        df = pd.DataFrame(np.hstack((face_data, labels)), columns=[str(i) for i in range(face_data.shape[1])] + ['name'])
        df.to_csv(self.save_path, index=False)
        print(f"Data prepared and saved to {self.save_path}")

if __name__ == "__main__":
    preparer = DataPreparer()
    preparer.prepare_data()
