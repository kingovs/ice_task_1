# main.py
from prepare_data import DataPreparer
from train_knn import KNNTrainer
from recognize_faces_from_images import FaceRecognizer

# To run the program from scratch, delete face_data.csv and knn_model.pkl, 
# then uncomment lines 13 to 23.
# then just execute this main.py. The folder structure should match the screenshot
# the download_cage.py code automatically downloads images of Cage's face,
# Just make sure that the images are sitting in the folders data/cage and data/others
# and not in subfolders therein
# Note.. download_cage.py would need to be run directly from that file

# # Prepare data
# data_preparer = DataPreparer()
# data_preparer.prepare_data()

# # Train KNN model
# knn_trainer = KNNTrainer()
# knn_trainer.train_knn()

# Recognize faces from images
face_recognizer = FaceRecognizer()
face_recognizer.recognize_faces()