
import subprocess
import sys
import socket
import time
import piDriverSafetyWebcam as driver
import config
import threading
import _thread
import json

path = config.homeDirectory + "src/resources/data.txt"

with open(path, "r") as myfile:     # Reading file for runParameter
    data = myfile.read()
    data = data.split(",")

runParameter = data[0]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 8081)
sock.bind(server_address)

sock.listen(5)

while True:

    newCamera = driver.camera()
    closing = [0,0]

    print("Waiting for Client...")
    connection, client_address = sock.accept()

    print("Connection from ", client_address)

    client_port = client_address[1]
    time.sleep(10)
    connection_object = {'parameters': {},
                         'face_detection': False,
                         'train': False,
                         'run_application': False}
    childThread1 = threading.Thread(target=driver.runSafety, args=(newCamera, connection, closing, 1))
    childThread0 = threading.Thread(target=driver.runSafety, args=(newCamera, connection, closing, 0))
    runThread = threading.Thread(target=driver.runSafety, args=(newCamera, connection, closing, 0))
    trainThread = threading.Thread(target=driver.trainDecisionTree, args=())
    newCamera.loadParameters()
    print("Completed loading. Please proceed.")

    if runParameter == 0:
        # Check for presence of face first
        driver.runSafety(newCamera, connection, closing, 0)

    while True:

        data = connection.recv(64)

        if data == b'':
            break

        else:

            if str(data)[2:18] == 'connection_found':
                closing.append(1)
                # Stops driver safety function
                closing = [0,0]
                runParameter = 1

            if str(data)[2:12] == 'parameters':

                print("Setting parameters : ")
                print(data)
                jargon = str(data)[12:]
                jargon = jargon.replace("\\x00", "") # Removing all null characters
                jargon = jargon.replace("\x00", "")
                param = jargon.split("*")            # Initializing list from string
                print("Parameters list: ", param)
                connection_object['parameters'] = param
                obj = {}
                # obj['profile_id'] = param[1]
                # obj['profile_name'] = param[2]
                obj['EYE_AR_THRESH'] = param[5]
                obj['debug'] = param[4]
                obj['alert_sensitivity'] = 1
                obj['detection_sensitivity'] = 1

                print("My saved object is : ", obj)

                cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/parameters.mp3"
                subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

                newCamera.setParameters(obj)

            if str(data)[2:9] == 'default':
                newCamera.setDefaultParameters()

            if str(data)[2:7] == 'train':
                trainThread.start()

            if str(data)[2:12] == 'stop_train':
                print("Interrupting training thread. You can train again, or fallback to default settings using the 'default' command.")
                trainThread.exit()

            if str(data)[2:16] == 'face_detection':
                print("Inside face detection")
                connection_object['face_detection'] = driver.face_detection(newCamera)

            if str(data)[2:6] == 'run1':

                # driver.runSafety(newCamera, connection)
                if childThread0.isAlive():
                    print("Exiting application. Set new parameters to start a new instance of the application.")
                    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/stop.mp3"
                    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                    childThread0.exit()

                if bool(connection_object.get('parameters')) == True and bool(connection_object.get('face_detection')) == True:
                    childThread1.start()
                else:
                    print("Please try again.")

            if str(data)[2:6] == 'run0':

                # driver.runSafety(newCamera, connection)
                if childThread1.isAlive():
                    print("Exiting application. Set new parameters to start a new instance of the application.")
                    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/stop.mp3"
                    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                    childThread1.exit()
                if bool(connection_object.get('parameters')) == True and bool(connection_object.get('face_detection')) == True:
                    childThread0.start()
                else:
                    print("Please try again.")

            if str(data)[2:6] == 'stop':
                closing.append(1)
                if childThread1.isAlive():
                    print("Exiting application. Set new parameters to start a new instance of the application.")
                    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/stop.mp3"
                    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                    childThread1.join()
                if childThread0.isAlive():
                    print("Exiting application. Set new parameters to start a new instance of the application.")
                    cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/stop.mp3"
                    subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                    childThread0.join()
                else:
                    print("Application is not running, invalid option.")
                closing = [0,0]

            if str(data)[2:6] == 'exit':
                cmd = config.homeDirectory + "sound/shell/playSound.sh " + config.homeDirectory + "sound/files/exit.mp3"
                subprocess.Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                _thread.exit()
                break

            # with open(path, "a") as myfile:
            #     myfile.write(str(data))
            # print("This is my data: ", data[1:3])
            # print(data)

myfile.close()
sock.close()
