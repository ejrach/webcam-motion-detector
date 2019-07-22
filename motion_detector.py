from cv2 import cv2
import time
import pandas
from datetime import datetime

first_frame = None
status_list = [None,None]   #prepopulate the status array so we can access the first and second items
times = []
df = pandas.DataFrame(columns=["Start","End"])

video = cv2.VideoCapture(0)

while True:
    
    # check: is to make sure the video is running
    # frame: is the video frame captures. This returns a numpy array we can loop through
    check, frame = video.read()

    # 0 = nothing detected
    status = 0

    # capture the current frame of the video into 'gray'
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)     #these are common values, reducing noise of the image

    # We want to capture an image of the first frame of the video capture.
    # If the first frame is capture, then return back to the beginning of the while loop
    if first_frame is None:
        first_frame = gray
        continue

    # finding the difference between the first static frame 'first_frame' 
    # and the current video frame 'gray'
    delta_frame = cv2.absdiff(first_frame,gray)

    # Now we are blacking out the pixels which are less than a 30 intensity (0 is black).
    # 255 is the max intensity (255 is white)
    # Then access the second item by using [1], which returns the actual frame
    thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    # smooth the threshold frame to remove black areas
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # now find the contours of the distinct objects in the image of the current frame
    cnts, hierarchy=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # If the contour has less than 1000 pixels, then skip, otherwise continue procecessing
        # the next lines.
        if cv2.contourArea(contour) < 10000:
            continue
        # 1 = something detected
        status = 1
        # draw a rectangle
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 3)

    status_list.append(status)

    status_list=status_list[-2:]

    #record when an object leaves the frame
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    #record when an object enters the frame
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    

    cv2.imshow("Gray frame",gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold frame", thresh_frame)
    cv2.imshow("Color frame", frame)

    key = cv2.waitKey(1)
    #print(gray)
    #print(delta_frame)

    if key==ord('q'):
        if status == 1:
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows