
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder
import cv2

def train(train_dir, n_neighbors = None, knn_algo = 'ball_tree', verbose=False):

    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    
    # Dump the trained decision tree classifier with Pickle
    model_save_path = '/home/pi/Desktop/model_classifier.pkl'
    # Open the file to save as pkl file
    with open(model_save_path, 'wb') as model_pkl:
        pickle.dump(knn_clf, model_pkl)
    # Close the pickle instance
    model_pkl.close()
    print("Model created successfully.")

    #return knn_clf

train("TrainData")