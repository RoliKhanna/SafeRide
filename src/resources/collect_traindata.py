from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy
import csv
import sys
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def collect_traindata(name):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
	headCascade = cv2.CascadeClassifier("resources/haarcascade_head.xml")
	faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_alt.xml")

	print("[INFO] camera sensor warming up...")

	vs = cv2.VideoCapture(0)
	vs.set(cv2.CAP_PROP_FPS, 8.0)
	time.sleep(2.0)
	inAttention = False
	lastAttentionFrame = 0;
	lastAttentionFrameThresh = 48;
	frameNumber = 0
	numFrames = 0

	trainCoordinates = []
	filePath = "resources/TrainData/pratik/" + name + ".csv"
	fid = open(filePath,"w")
	out = csv.writer(fid, delimiter=',',quoting=csv.QUOTE_ALL)

	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	endTime = time.time() + 30		# 10 seconds

	# loop over the frames from the video stream
	while numFrames < 200:
	    # grab the frame from the threaded video stream, resize it to
	    # have a maximum width of 400 pixels, and convert it to
	    # grayscale

		if time.time() > endTime:
			print("Training completed!")
			break

		ret, frame = vs.read()
		numFrames = numFrames + 1
		frame = imutils.resize(frame, width=300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frameNumber = frameNumber + 1;

	    # detect faces in the grayscale frame
	    # rects = detector(gray, 0)

		faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(200,200), flags=cv2.CASCADE_SCALE_IMAGE)

		if (len(faces) > 0):
			for (x, y, w, h) in faces:
				rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				ns = [];
				for (xs, ys) in shape:
					cv2.circle(frame, (xs, ys), 1, (0, 255, 0), -1)
					ns.append(xs - x)
					ns.append(ys - y)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ns.append(leftEAR)
				ns.append(rightEAR)
				trainCoordinates.append(ns)
				out.writerow(ns)
		else:
			rects = detector(gray, 0)
			if (len(rects) > 0):
				for rect in rects:
					(x, y, w, h) = face_utils.rect_to_bb(rect)
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
					shape = predictor(gray, rect)
					shape = face_utils.shape_to_np(shape)
					ns = [];
					for (xs, ys) in shape:
						cv2.circle(frame, (xs, ys), 1, (0, 0, 255), -1)
						ns.append(xs - x)
						ns.append(ys - y)
					leftEye = shape[lStart:lEnd]
					rightEye = shape[rStart:rEnd]
					leftEAR = eye_aspect_ratio(leftEye)
					rightEAR = eye_aspect_ratio(rightEye)
					ns.append(leftEAR)
					ns.append(rightEAR)
					trainCoordinates.append(ns)
					out.writerow(ns)
			else:
				heads = headCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(200,200), flags=cv2.CASCADE_SCALE_IMAGE )
				if (len(heads) > 0):
					for (x, y, w, h) in heads:
						rect = dlib.rectangle(int(x), int(y + 0.3*h), int(x+w), int(y+h))
						cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
						shape = predictor(gray, rect)
						shape = face_utils.shape_to_np(shape)
						ns = [];
						for (xs, ys) in shape:
							cv2.circle(frame, (xs, ys), 1, (255, 0, 0), -1)
							ns.append(xs - x)
							ns.append(ys - y)
						leftEye = shape[lStart:lEnd]
						rightEye = shape[rStart:rEnd]
						leftEAR = eye_aspect_ratio(leftEye)
						rightEAR = eye_aspect_ratio(rightEye)
						ns.append(leftEAR)
						ns.append(rightEAR)
						trainCoordinates.append(ns)
						out.writerow(ns)
		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(50) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	fid.close()

# eyes_closed
# eyes_open
# looking_down
# looking_up
# looking_left
# looking_right
# normal

# collect_traindata("normal")
