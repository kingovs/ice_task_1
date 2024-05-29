import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

class FaceRecognizer:
    def __init__(self, image_dir='test_images', model_path='knn_model.pkl'):
        self.image_dir = image_dir
        self.model_path = model_path

    def recognize_faces(self):
        knn = joblib.load(self.model_path)
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for image_name in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray, 1.5, 5)
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
                name = knn.predict(face_img)[0]
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Recognize Face")
            plt.show()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
