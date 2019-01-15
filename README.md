# SafeRide
Safety feature for vehicles on a Raspberry Pi

# README #

Step 1: Make sure all the prerequisites are installed: OpenCV, Python, dlib, face_recognition

OpenCV and Python installation: https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/

dlib installation: https://www.pyimagesearch.com/2017/05/01/install-dlib-raspberry-pi/

face_recognition installation: pip3 install face_recognition

Step 2: Clone the repo in your system.

Step 3: To enable audio at boot:

$ sudo nano /etc/rc.local

Append the following line at the end of this file (before exit 0), and save it:

amixer cset numid=3 1

Step 4: To have the script run constantly in the background, type the following commands in terminal:

$ crontab -e

Append the following line at the end of this file, and save it:

*/1 * * * * . /yourLocalPath/airide/commands.sh

Step 5: Make the changes in the config.py file with your local paths in /src/config.py

Step 6: Skip step 4 in case you want to run the script explicitly, run it as given below, in the root directory:

$ cd src

$ python serverApp.py

$ ./backend
