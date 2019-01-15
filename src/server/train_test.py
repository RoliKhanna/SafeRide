from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
# from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial import distance as dist
import pickle
import config

testPath = config.homeDirectory + "/src/test/DT/standard/"

def getTime():
    currentTime = time.localtime()
    hour = str(currentTime.tm_hour)
    minute = str(currentTime.tm_min)
    second = str(currentTime.tm_sec)
    result = hour + ":" + minute + ":" + second
    return result

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    # A = dist.euclidean(eye[1], eye[5])
    # B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    # C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    # ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    # return ear

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def prepare_dataset():
    print("Preparing new decision tree")
    ################################################# Train Facial Landmark Model with Data Collected ###########################################
    dataset1 = np.loadtxt('resources/TrainData/combined/normal.csv', delimiter=',', dtype=np.str)
    dataset1 = [[y.replace('"','') for y in x] for x in dataset1]
    dataset1 = np.asarray(dataset1)
    dataset1 = dataset1.astype(np.float)
    labels1 = np.ones(dataset1.shape[0])
    dataset2 = np.loadtxt('resources/TrainData/combined/looking_down.csv', delimiter=',', dtype=np.str)
    labels2 = 2*np.ones(dataset2.shape[0])
    # labels2 = np.ones(dataset2.shape[0])
    dataset2 = [[y.replace('"','') for y in x] for x in dataset2]
    dataset2 = np.asarray(dataset2)
    dataset2 = dataset2.astype(np.float)

    dataset3 = np.loadtxt('resources/TrainData/combined/looking_left.csv', delimiter=',', dtype=np.str)
    labels3 = 3*np.ones(dataset3.shape[0])
    # labels3 = np.ones(dataset3.shape[0])
    dataset3 = [[y.replace('"','') for y in x] for x in dataset3]
    dataset3 = np.asarray(dataset3)
    dataset3 = dataset3.astype(np.float)

    dataset4 = np.loadtxt('resources/TrainData/combined/looking_right.csv', delimiter=',', dtype=np.str)
    labels4 = 4*np.ones(dataset4.shape[0])
    # labels4 = np.ones(dataset4.shape[0])
    dataset4 = [[y.replace('"','') for y in x] for x in dataset4]
    dataset4 = np.asarray(dataset4)
    dataset4 = dataset4.astype(np.float)

    dataset5 = np.loadtxt('resources/TrainData/combined/looking_up.csv', delimiter=',', dtype=np.str)
    labels5 = 5*np.ones(dataset5.shape[0])
    # labels5 = np.ones(dataset5.shape[0])
    dataset5 = [[y.replace('"','') for y in x] for x in dataset5]
    dataset5 = np.asarray(dataset5)
    dataset5 = dataset5.astype(np.float)

    dataset6 = np.loadtxt('resources/TrainData/combined/eyes_closed.csv', delimiter=',', dtype=np.str)
    labels6 = 6*np.ones(dataset6.shape[0])
    # labels6 = np.ones(dataset6.shape[0])
    dataset6 = [[y.replace('"','') for y in x] for x in dataset6]
    dataset6 = np.asarray(dataset6)
    dataset6 = dataset6.astype(np.float)
    #dataset7 = np.loadtxt('TrainData/mouth_open_yawn.csv', delimiter=',', dtype=np.str)
    #labels7 = 7*np.ones(dataset7.shape[0])
    #dataset7 = [[y.replace('"','') for y in x] for x in dataset7]
    #dataset7 = np.asarray(dataset7)
    #dataset7 = dataset7.astype(np.float)

    datasetCombined = np.concatenate((dataset1,dataset2,dataset3,dataset4,dataset5))
    labelsCombined = np.concatenate((labels1,labels2,labels3,labels4,labels5))
    X_train, X_test, y_train, y_test = train_test_split(datasetCombined, labelsCombined, test_size=0.3, random_state=42)
    # clf = tree.DecisionTreeClassifier()
    # clf = sklearn.ensemble.GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    clf = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10)
    clf.fit(X_train,y_train)
    y_test_predict = clf.predict(X_test)
    # print(accuracy_score(y_test, y_test_predict))

    # Dump the trained decision tree classifier with Pickle
    decision_tree_pkl = 'classifier.pkl'
    # Open the file to save as pkl file
    decision_tree_model_pkl = open(decision_tree_pkl, 'wb')
    pickle.dump(clf, decision_tree_model_pkl)
    # Close the pickle instances
    decision_tree_model_pkl.close()
    print("Model created successfully.")
    return clf

def begin_classifier():

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('DT.avi', fourcc, 4.0, (200,112))

    option = input("Use pre-trained model(1) or train again(2) ? ")
    if option == 1:
        # Loading the saved decision tree model pickle
        decision_tree_pkl = 'resources/decision_tree_classifier.pkl'
        decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
        clf = pickle.load(decision_tree_model_pkl)
    else:
        clf = prepare_dataset()     # Decision tree model

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
    headCascade = cv2.CascadeClassifier("resources/haarcascade_head.xml")
    faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_alt.xml")

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0)
    vs.set(cv2.CAP_PROP_FPS, 8.0)
    time.sleep(2.0)
    inAttention = False
    lastAttentionFrame = 0
    lastAttentionFrameThresh = 48
    frameNumber = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    xlast = 0
    ylast = 0
    hlast = 0
    wlast = 0
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        ret, frame = vs.read()
        #framecrop = frame
        frame = imutils.resize(frame, width=200)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameNumber = frameNumber + 1
        # detect faces in the grayscale frame
        # rects = detector(gray, 0)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(200,200), flags=cv2.CASCADE_SCALE_IMAGE )
        if (len(faces) > 0):
            for (x, y, w, h) in faces:
                xlast = x
                ylast = y
                hlast = h
                wlast = w
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                ns = []
                for (xs, ys) in shape:
                    # cv2.circle(frame, (xs, ys), 1, (0, 255, 0), -1)
                    ns.append(xs - x)
                    ns.append(ys - y)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ns.append(leftEAR)
                ns.append(rightEAR)
                ns = np.asarray(ns)
                ys = clf.predict([ns])
                if (leftEAR + rightEAR < 0.5):
                    cv2.putText(frame, "EAR EYES CLOSED", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("EAR EYES CLOSED")
                if (ys == 1.0):
                    cv2.putText(frame, "NORMAL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                elif(ys == 2.0):
                    cv2.putText(frame, "Looking DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("LOOKING DOWN")
                elif(ys == 3.0):
                    cv2.putText(frame, "Looking LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("LOOKING LEFT")
                elif (ys == 4.0):
                    cv2.putText(frame, "Looking RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("LOOKING RIGHT")
                elif (ys == 5.0):
                    cv2.putText(frame, "Looking UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("LOOKING UP")
                elif (ys == 6.0):
                    cv2.putText(frame, "DT EYES CLOSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("DT EYES CLOSED")
                elif (ys == 7.0):
                    cv2.putText(frame, "YAWNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print("YAWNING")
        else:
            graycrop = gray[ylast - int(0.5*hlast):ylast + int(1.5*hlast), xlast - int(0.5*wlast): xlast + int(1.5*wlast)].copy()
            framecrop = frame[ylast - int(0.5*hlast):ylast + int(1.5*hlast), xlast - int(0.5*wlast): int(xlast+1.5*wlast)].copy()
            rects = detector(graycrop, 0)
            if (len(rects) > 0):
                for rect in rects:
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(framecrop, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    shape = predictor(graycrop, rect)
                    shape = face_utils.shape_to_np(shape)
                    ns = []
                    for (xs, ys) in shape:
                        # cv2.circle(framecrop, (xs, ys), 1, (0, 0, 255), -1)
                        ns.append(xs - x)
                        ns.append(ys - y)
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ns.append(leftEAR)
                        ns.append(rightEAR)
                        ns = np.asarray(ns)
                        ys = clf.predict([ns])
                        if (leftEAR + rightEAR < 0.5):
                            cv2.putText(frame, "EYES CLOSED using EAR", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("EYES CLOSED USING EAR")
                        if (ys == 1.0):
                            cv2.putText(frame, "NORMAL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("NORMAL MODE")
                        elif (ys == 2.0):
                            cv2.putText(frame, "Looking DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking DOWN")
                        elif (ys == 3.0):
                            cv2.putText(frame, "Looking LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking LEFT")
                        elif (ys == 4.0):
                            cv2.putText(frame, "Looking RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking RIGHT")
                        elif (ys == 5.0):
                            cv2.putText(frame, "Looking UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking UP")
                        elif (ys == 6.0):
                            cv2.putText(frame, "EYES CLOSED using DT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("EYES CLOSED using DT")
                        elif (ys == 7.0):
                            cv2.putText(frame, "YAWNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("YAWNING")
            else:
                profile = cv2.CascadeClassifier(config.casc3).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)

                if len(profile) > 0:
                    # I detected a right profile
                    for (x, y, w, h) in profile:
                        xlast = x
                        ylast = y
                        hlast = h
                        wlast = w
                        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        ns = []
                        for (xs, ys) in shape:
                            # cv2.circle(frame, (xs, ys), 1, (0, 255, 0), -1)
                            ns.append(xs - x)
                            ns.append(ys - y)
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ns.append(leftEAR)
                        ns.append(rightEAR)
                        ns = np.asarray(ns)
                        ys = clf.predict([ns])
                        if (leftEAR + rightEAR < 0.5):
                            cv2.putText(frame, "EYES CLOSED using EAR", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("EYES CLOSED USING EAR")
                        if (ys == 1.0):
                            cv2.putText(frame, "NORMAL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("NORMAL MODE")
                        elif (ys == 2.0):
                            cv2.putText(frame, "Looking DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking DOWN")
                        elif (ys == 3.0):
                            cv2.putText(frame, "Looking LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking LEFT")
                        elif (ys == 4.0):
                            cv2.putText(frame, "Looking RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking RIGHT")
                        elif (ys == 5.0):
                            cv2.putText(frame, "Looking UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("Looking UP")
                        elif (ys == 6.0):
                            cv2.putText(frame, "EYES CLOSED using DT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("EYES CLOSED using DT")
                        elif (ys == 7.0):
                            cv2.putText(frame, "YAWNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                            print("YAWNING")

                else:
                    # Flip frame, check for left profile
                    flippedFrame = cv2.flip(gray, 1)    # horizontal flip
                    profile = cv2.CascadeClassifier(config.casc3).detectMultiScale(flippedFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(profile):
                        # I detected a left profile
                        for (x, y, w, h) in profile:
                            xlast = x
                            ylast = y
                            hlast = h
                            wlast = w
                            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            ns = []
                            for (xs, ys) in shape:
                                # cv2.circle(frame, (xs, ys), 1, (0, 255, 0), -1)
                                ns.append(xs - x)
                                ns.append(ys - y)
                            leftEye = shape[lStart:lEnd]
                            rightEye = shape[rStart:rEnd]
                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)
                            ns.append(leftEAR)
                            ns.append(rightEAR)
                            ns = np.asarray(ns)
                            ys = clf.predict([ns])
                            if (leftEAR + rightEAR < 0.5):
                                cv2.putText(frame, "EYES CLOSED using EAR", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("EYES CLOSED USING EAR")
                            if (ys == 1.0):
                                cv2.putText(frame, "NORMAL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("NORMAL MODE")
                            elif (ys == 2.0):
                                cv2.putText(frame, "Looking DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("Looking DOWN")
                            elif (ys == 3.0):
                                cv2.putText(frame, "Looking LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("Looking LEFT")
                            elif (ys == 4.0):
                                cv2.putText(frame, "Looking RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("Looking RIGHT")
                            elif (ys == 5.0):
                                cv2.putText(frame, "Looking UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("Looking UP")
                            elif (ys == 6.0):
                                cv2.putText(frame, "EYES CLOSED using DT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("EYES CLOSED using DT")
                            elif (ys == 7.0):
                                cv2.putText(frame, "YAWNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                                print("YAWNING")

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Saving image for testing
        # timestamp = getTime()
        # myPath = testPath + timestamp + ".jpg"
        # cv2.imwrite(myPath, frame)

        out.write(frame)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

begin_classifier()
