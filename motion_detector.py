from cv2 import cv2
import time

video = cv2.VideoCapture(0)

a=1

while True:
    a = a+1
    
    # check: is to make sure the video is running
    # frame: is the video frame captures. This returns a numpy array we can loop through
    check, frame = video.read()

    print(check)
    print(frame)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # time.sleep(3)
    cv2.imshow("Capturing",gray)

    key = cv2.waitKey(1)

    if key==ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows