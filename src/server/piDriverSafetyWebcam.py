
activate_this_file = "/home/pi/.virtualenvs/cv/bin/activate_this.py"

with open(activate_this_file) as f:
    code = compile(f.read(), activate_this_file, 'exec')
    exec(code, dict(__file__=activate_this_file))

import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
from sklearn import neighbors
import pickle
from os import listdir
from os.path import isdir, join, isfile, splitext
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder
import imutils
import argparse
import numpy as np
import dlib
import subprocess
import time
import socket
import json
import config
# import collect_traindata as train
# import train_test as test

path = config.homeDirectory + "/src/resources/checkClient.txt"
testPath = config.homeDirectory + "/src/test/roli/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class camera:

    '''Contains algorithm and camera settings'''

    cascPath1 = config.casc1 # Face cascade
    cascPath2 = config.casc2 # Head cascade
    cascPath3 = config.casc3 # Profile cascade
    shape_predictor = config.shape
    faceCascade = cv2.CascadeClassifier(cascPath1)
    headCascade = cv2.CascadeClassifier(cascPath2)
    profileCascade = cv2.CascadeClassifier(cascPath3)
    predictor = dlib.shape_predictor(shape_predictor)

    profile_id = 1
    profile_name = "ABC"
    EYE_AR_THRESH = 0.3             # default values
    MOUTH_THRESH = 0.6
    FPS = 8.0
    DROWSY_PERCENTAGE = 0.8
    INATT_PERCENTAGE = 0.8
    YAWN_PERCENTAGE = 0.8
    drowsyCheck = True
    inattCheck = True
    debug = False
    detection_sensitivity = 1       # 1 for high, 0 for low
    alert_frequency = 5             # alerts will be spaced in 5 second intervals, default

    def loadParameters(self):
        # Loading data from JSON file
        path = config.homeDirectory + '/src/resources/data.json'
        with open(path) as json_file:
            data = json.load(json_file)
            # print("JSON Data: ", data)
            self.profile_id = str(data['profile_id'])
            self.profile_name = str(data['profile_name'])
            self.EYE_AR_THRESH = float(data['camera_settings']['EYE_AR_THRESH'])
            self.drowsyCheck = data['camera_settings']['drowsyCheck']
            self.inattCheck = data['camera_settings']['inattentionCheck']
            self.debug = data['camera_settings']['debug']
            self.detection_sensitivity = data['camera_settings']['detection_sensitivity']
            self.alert_sensitivity = data['camera_settings']['alert_sensitivity']

    def setParameters(self, param):
        path = config.homeDirectory + '/src/resources/data.json'
        data = {}
        data['profile_id'] = self.profile_id
        data['profile_name'] = self.profile_name
        data['camera_settings'] = param
        # Writing dictionary data to JSON file
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def setDefaultParameters(self):
        self.EYE_AR_THRESH = 0.3
        self.MOUTH_THRESH = 0.6
        self.debug = False
        self.detection_sensitivity = 1
        self.alert_sensitivity = 1

def predict(X_img_path, knn_clf = None, model_save_path ="", DIST_THRESH = .5):

    '''
    recognizes faces in given image, based on a trained knn classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    '''

    if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]

def predict(X_img, knn_clf = None, model_save_path ="", DIST_THRESH = .5):

    '''
    Recognizes faces in given image, based on a trained knn classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    '''

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either through knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []
    t1=time.time()
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)
    t2 =  time.time()
    print(str(t2-t1) + " time taken to get feature vector")

    t1=time.time()
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    t2 =  time.time()
    print(str(t2-t1) + " time taken for KNN")

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]


def eye_aspect_ratio(eye):

    '''
    Calculates and returns the EAR value for drowsiness
    :param eye: eye descriptor values
    :return: EAR value
    '''

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):

    '''
    Calculates and returns the MAR value for yawning
    :param mouth: mouth descriptor values
    :return: EAR value
    '''

    A = np.linalg.norm(mouth[3] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[10])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B)/(2*C)
    return mar

def getTime():
    currentTime = time.localtime()
    hour = str(currentTime.tm_hour)
    minute = str(currentTime.tm_min)
    second = str(currentTime.tm_sec)
    result = hour + ":" + minute + ":" + second
    return result

def face_detection(cam):

    '''
    Checks the presence of a face in the frame
    for five seconds. Required metric for ensuring
    the driver is visible and identifiable by the
    application.
    :param cam: camera object
    :return: True/False, depending on whether face
    is detected or not
    '''

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    fps = FPS().start()
    frameNumber = 0
    vs = cv2.VideoCapture(0)
    my_fps = cam.FPS
    vs.set(cv2.CAP_PROP_FPS, my_fps)   #setting to fps = 8
    face_count = 0

    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/face.mp3"
    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    print('Beginning face detection in 5 seconds: ')
    time.sleep(5.0)
    t_end = time.time() + 5     # checking for 5 seconds

    print("Starting face detection")
    while True:

        if time.time() > t_end:
            break

        else:
            ret, frame = vs.read()
            frame = imutils.resize(frame, width=200)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cam.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE )
            if len(faces) > 0 :
                face_count = face_count + 1
            fps.update()
            frameNumber = frameNumber + 1

            if cam.debug == True:
                cv2.imshow('Video', frame)

    fps.stop()
    cv2.destroyAllWindows()

    if face_count > 5:
        cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/face_success.mp3"
        subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
        print("Face detected successfully!")
        return True
    else:
        cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/face_no_success.mp3"
        subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
        print("Face detection unsuccessful, try again!")
        return False

def trainDecisionTree():

    '''
    Train a decision tree for each user,
    and customise the parameter values according
    to each user's compatible values.
    '''

    # engine = pyttsx.init()
    time.sleep(5.0)
    t_end = time.time() + 5     # training for 5 seconds
    i = 0

    while True:
        if time.time() == t_end:
            i+=1
            t_end = time.time() + 5
        if i == 0:
            # engine.say('Please keep your eyes open for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("eyes_open")
        elif i == 1:
            # engine.say('Please keep your eyes closed for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("eyes_closed")
        elif i == 2:
            # engine.say('Please look down for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("looking_down")
        elif i == 3:
            # engine.say('Please look up for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("looking_up")
        elif i == 4:
            # engine.say('Please look right for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("looking_right")
        elif i == 5:
            # engine.say('Please look left for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("looking_left")
        elif i == 1:
            # engine.say('Please look at the camera for 10 seconds.')
            # engine.runAndWait()
            train.collect_traindata("Normal")
        else:
            print("Training completed successfully!")
            break

def train(cam):

    '''
    A training approach to fine-tune the EAR
    value for each user.
    :param cam: camera object
    '''

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    fps = FPS().start()
    frameNumber = 0
    vs = cv2.VideoCapture(0)
    my_fps = cam.FPS
    vs.set(cv2.CAP_PROP_FPS, my_fps)
    eyes_open = 0
    eyes_close = 0

    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/train1.mp3"
    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    print('Beginning training for open eyes in 5 seconds: ')
    time.sleep(5.0)
    t_end = time.time() + 5     # training for 5 seconds

    sample_ears = []

    while True:

        if time.time() > t_end:
            break
        else:
            ret, frame = vs.read()
            frame = imutils.resize(frame, width=200)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cam.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE )
            #print(faces)
            fps.update()
            frameNumber = frameNumber + 1

            for (x, y, w, h) in faces:

                dlibrect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                shape = cam.predictor(gray, dlibrect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                sample_ears.append(ear)
                eyes_open = sum(sample_ears)/len(sample_ears)
                # print("My EAR threshold: ", cam.EYE_AR_THRESH)

            if cam.debug == True:
                cv2.imshow('Video', frame)

    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/train2.mp3"
    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    print('Beginning training for closed eyes in 5 seconds: ')
    time.sleep(5.0)

    t_end = time.time() + 5     # training for 5 seconds

    sample_ears = []

    while True:

        if time.time() > t_end:
            break
        else:
            ret, frame = vs.read()
            frame = imutils.resize(frame, width=200)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cam.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE )
            #print(faces)
            fps.update()
            frameNumber = frameNumber + 1

            for (x, y, w, h) in faces:

                dlibrect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                shape = cam.predictor(gray, dlibrect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                sample_ears.append(ear)
                eyes_close = sum(sample_ears)/len(sample_ears)
                print("My EAR threshold: ", cam.EYE_AR_THRESH)

            if cam.debug == True:
                cv2.imshow('Video', frame)

    fps.stop()
    cv2.destroyAllWindows()

    cam.EYE_AR_THRESH = (eyes_open + eyes_close)/2.0    # Average, for now
    print("Training completed. The new EAR is : ", cam.EYE_AR_THRESH)

def faceRec():

    '''
    Recognizes faces of them people. Huehuehue.
    '''

    # Loading models for face recognition
    model_pkl_name = config.homeDirectory + 'src/resources/model_classifier.pkl'
    model_pkl = open(model_pkl_name, 'rb')
    knn_clf = pickle.load(model_pkl)

    # Face recognition related parameters
    personCount = 0
    recognitionThreshold = 25
    reci = 0                   # Face recognition iterator
    frames = []                # Frames for face recognition
    faceIdentified = "No one"  # Default

    fps = FPS().start()
    vs = cv2.VideoCapture(0)
    my_fps = cam.FPS
    vs.set(cv2.CAP_PROP_FPS, my_fps)

    while True:

        ret, frame = vs.read()
        frame = imutils.resize(frame, width=200)

        if personCount > 120:    # Checking absence for 120 frames
            reci = 0

        if reci == recognitionThreshold:
            people = list(set(frames))
            #print("People identified : ", people)
            counter = []
            found = 0

            for i in range(0, len(people)):
                counter.append(frames.count(people[i])/len(frames))
                if counter[i] > max(counter):
                    faceIdentified = counter[i]
                    found = i

            if counter[found] > 0.50:  # thresholding to 50% detection
                faceIdentified = people[found]
                print("Successfully identified " + faceIdentified)
                reci = recognitionThreshold + 1
                personCount += 1        # I think

            else:
                faceIdentified = "No one"
                print("Couldn't identify anyone. Trying again.")
                reci = 0  # starting face recognition again

        if reci < recognitionThreshold:
            print("Face recognition in progress....")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Find all the faces and face encodings in the frame of video
            preds = predict(rgb_frame, knn_clf=knn_clf)
            for pred in preds:
                loc = pred[1]
                name = pred[0]
                frames.append(name)

            reci += 1
            continue

def checkQueuePercentage(check, drowsyMaxSize, inattMaxSize):

    '''
    The alerting mechanism is based on a Queue system,
    wherein the most recent 'n' frames are accounted, the
    older ones are de-queued. The queue elements are checked
    for 'D' (drowsy frame), 'I' (inattention frame), 'L'
    (left inattention frame), and 'R' (right inattention
    frame). The percentage of each is calculated and
    returned.
    :param check: Queue with frame values
    :param drowsyMaxSize: Maximum drowsy frames (for splitting)
    :param inattentionMaxSize: Maximum inattention frames (for splitting)
    :return: percentage composition of each alert in the queue
    '''

    drowsyCount = 0
    inattCount = 0
    rightCount = 0
    leftCount = 0

    drowsyCheck = check[-drowsyMaxSize:]
    inattCheck = check[-inattMaxSize:]

    for i in range(0, len(drowsyCheck)):
        if drowsyCheck[i] == "D":
            drowsyCount += 1

    for i in range(0, len(inattCheck)):
        if inattCheck[i] == "I":
            inattCount += 1
        elif inattCheck[i] == "L":
            leftCount += 1
        elif inattCheck[i] == "R":
            rightCount += 1

    drowsyPercentage = drowsyCount/len(drowsyCheck)
    inattPercentage = inattCount/len(inattCheck)
    rightPercentage = rightCount/len(inattCheck)
    leftPercentage = leftCount/len(inattCheck)

    return [drowsyPercentage, inattPercentage, leftPercentage, rightPercentage]

def runSafety(cam, connection, closing, sensitivity):
# def runSafety(cam,sensitivity,closing):
# 1: high, 0:low sensitivity

    '''
    This is the primary 'run' function, it runs the
    main driver safety algorithm logic. It is connected
    to the backendApp, and sends back alerts to it in
    the form of socket messages.
    :param cam: camera object
    :param connection: socket object shared with serverApp
    :param closing: list object shared with serverApp to
    handle termination
    :param sensitivity: threshold decision for drowsiness/
    inattention
    '''

    # Sound prompt to inform user of the status
    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/run.mp3"
    subprocess.Popen([cmd], shell = True, stdin = None, stdout = None, stderr = None, close_fds = True)

    # Camera and algorithm related parameters
    cam.debug = True
    xlast = 0
    ylast = 0
    hlast = 0
    wlast = 0
    message = ""
    myQueue = []                # Queue for frame-wise alerts
    drowsyTimeCheck = time.time()
    inattTimeCheck = time.time()

    # high sensitivity: 4(inatt), 8(drowsy); medium sensitivity: 8(inatt), 16(drowsy)
    if sensitivity == 0:
        # low sensitivity
        drowsyMaxSize = 20 # 16
        inattMaxSize = 8   # 8

    else:
        # high sensitivity
        drowsyMaxSize = 8
        inattMaxSize = 4

    # Setting video related parameters
    detector = dlib.get_frontal_face_detector()
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 4.0, (200,112))
    fps = FPS().start()
    frameNumber = 0
    vs = cv2.VideoCapture(0)
    my_fps = cam.FPS
    vs.set(cv2.CAP_PROP_FPS, my_fps)

    # Automatic adjustment to user's eye profile
    closeEye = []
    openEye = []
    trackEye = [0.30]
    diffEye = 0.08
    evaluation = True
    drowsyEAR = 0
    notDrowsyEAR = 0
    startTime = int(time.time())

    time.sleep(2.0)

    print("Starting the Driver Safety application ...")

    while True:

        # Capture frame-by-frame
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=200)

        sumThread = 0  # to handle thread stopping case
        for i in range(0, len(closing)):
            sumThread = sumThread + closing[i]

        if sumThread > 0:
            break

        detected = False
        cam.detection_sensitivity = 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(bright, [4], None, [256], [0,256])   # using channel 'v' or [4]

        faces = cam.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)
        fps.update()
        frameNumber = frameNumber + 1

        if len(faces) > 0:

            personCount = 0

            for (x, y, w, h) in faces:

                xlast = x
                ylast = y
                hlast = h
                wlast = w

                dlibrect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                shape = cam.predictor(gray, dlibrect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mouth = shape[mStart:mEnd]
                MAR = mouth_aspect_ratio(mouth)
                # Average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # print("EAR value: ", ear)
                # Automatic custom-EAR calculation

                if startTime + 2 < time.time() and evaluation == True:
                    tracking = trackEye[-1] - ear
                    if tracking >= diffEye:
                        closeEye.append(ear)
                    else:
                        openEye.append(ear)
                    trackEye.append(ear)

                if startTime + 40 < time.time() and evaluation == True:     # changed from 20 to 40
                    if len(closeEye) > 3 and len(openEye) > 3:  # minimum length of list
                        evaluation = False
                        drowsyEAR = sum(closeEye)/len(closeEye)
                        notDrowsyEAR = sum(openEye)/len(openEye)
                        # print("All EAR values: ", trackEye)
                        # print("Drowsy list: ", closeEye)
                        # print("Non drowsy list: ", openEye)
                        print("Drowsy average EAR: ", drowsyEAR)
                        print("Not drowsy average EAR: ", notDrowsyEAR)

                        if notDrowsyEAR < 0.28:     # Deciding factor
                            cam.EYE_AR_THRESH = 0.25
                        if drowsyEAR > 0.30:
                            cam.EYE_AR_THRESH = 0.32
                    else:
                        print("I need more EAR values. Not cool.")

                if ear < cam.EYE_AR_THRESH:
                    myQueue.append("D")
                    detected = True

                if MAR > cam.MOUTH_THRESH:
                    myQueue.append("D")
                    detected = True

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                # for (s, t) in shape:
                #     cv2.circle(frame, (s, t), 1, (0, 0, 255), -1)

        else:
            # Checking for profile faces here
            if cam.detection_sensitivity == 1:
                myQueue.append("I")
                personCount = 0
                # detected = True

            else:
                profile = cam.profileCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)
                # detected = False

                if len(profile) > 0:
                    # I detected a left profile
                    myQueue.append("L")
                    detected = True
                    personCount = 0
                    for (x, y, w, h) in profile:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                else:
                    # Flip frame, check for right profile
                    flippedFrame = cv2.flip(gray, 1)    # horizontal flip
                    profile = cam.profileCascade.detectMultiScale(flippedFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(profile):
                        # I detected a right profile
                        myQueue.append("R")
                        detected = True
                        personCount = 0
                        for (x, y, w, h) in profile:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if detected == False :

                graycrop = gray[ylast - int(0.5*hlast):ylast + int(1.5*hlast), xlast - int(0.5*wlast): xlast + int(1.5*wlast)].copy()
                framecrop = frame[ylast - int(0.5*hlast):ylast + int(1.5*hlast), xlast - int(0.5*wlast): int(xlast+1.5*wlast)].copy()
                rects = detector(graycrop, 0)

                if (len(rects) > 0):

                    for rect in rects:
                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        cv2.rectangle(framecrop, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        shape = cam.predictor(graycrop, rect)
                        shape = face_utils.shape_to_np(shape)
                        for (xs, ys) in shape:
                            cv2.circle(framecrop, (xs, ys), 1, (0, 0, 255), -1)
                            leftEye = shape[lStart:lEnd]
                            rightEye = shape[rStart:rEnd]
                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)
                            if (leftEAR + rightEAR < 0.5):
                                detected = True
                                myQueue.append("D")

                else:
                    # Use head detection here instead, landmark framing
                    head = cam.headCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(120,120), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(head) > 0:
                        for (x, y, w, h) in head:
                            dlibrect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Saving image for testing
        # timestamp = getTime()
        # myPath = testPath + timestamp + ".jpg"
        # cv2.imwrite(myPath, frame)

        # Writing frame into video
        out.write(frame)

        if detected == False:
            # Calculating brightness using histograms
            myQueue.append("0")
            brightness = 0
            for i in hist:
                if i[0] > 0:   # thresholding brightness value to 0, check
                    brightness += 1

            # print("Brightness : ", brightness)
            # print("Length of histogram array : ", len(hist))

            brightness = (brightness/len(hist))*100  # brightness percentage

            if brightness > 80:
                print("The frame is too bright. Brightness percentage: ", brightness)
            elif brightness < 20:
                print("The frame is too dark. Brightness percentage: ", brightness)

        if frameNumber > drowsyMaxSize and frameNumber > inattMaxSize:
            check = checkQueuePercentage(myQueue, drowsyMaxSize, inattMaxSize)

            if check[0] >= cam.DROWSY_PERCENTAGE:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                timestamp = getTime()
                newMessage = "111 Drowsiness detected at " + timestamp + "hrs."
                if drowsyTimeCheck + cam.alert_frequency < time.time():
                    drowsyTimeCheck = time.time()
                    print(newMessage)
                    if cam.drowsyCheck == True and cam.debug == False:
                        connection.send(newMessage.encode())
                detected = True

            if check[1] >= cam.INATT_PERCENTAGE:
                cv2.putText(frame, "INATTENTION DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                timestamp = getTime()
                newMessage = "333 Inattention detected at " + timestamp + "hrs."
                if inattTimeCheck + cam.alert_frequency < time.time():
                    inattTimeCheck = time.time()
                    print(newMessage)
                    if cam.inattCheck == True and cam.debug == False:
                        connection.send(newMessage.encode())
                detected = True

            if check[2] >= cam.INATT_PERCENTAGE:
                cv2.putText(frame, "INATTENTION DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                timestamp = getTime()
                newMessage = "333 Right Inattention detected at " + timestamp + "hrs."
                if inattTimeCheck + cam.alert_frequency < time.time():
                    inattTimeCheck = time.time()
                    print(newMessage)
                    if cam.inattCheck == True and cam.debug == False:
                        connection.send(newMessage.encode())
                detected = True

            if check[3] >= cam.INATT_PERCENTAGE:
                cv2.putText(frame, "INATTENTION DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                timestamp = getTime()
                newMessage = "333 Left Inattention detected at " + timestamp + "hrs."
                if inattTimeCheck + cam.alert_frequency < time.time():
                    inattTimeCheck = time.time()
                    print(newMessage)
                    if cam.inattCheck == True and cam.debug == False:
                        connection.send(newMessage.encode())
                detected = True

            myQueue.pop(0)

        # Display the resulting frame
        if cam.debug == True:
            cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    cv2.destroyAllWindows()

# cam = camera()
# runSafety(cam,0,closing)
# train(cam)
# face_detection(cam)
