import cv2
import time
import numpy as np

fourcc=cv2.VideoWriter_fourcc(*"XVID")
outputfile=cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))

cam=cv2.VideoCapture(0)
time.sleep(0)
bg=0
for i in range(60):
    ret,bg=cam.read()

bg=np.flip(bg,axis=1)
while cam.isOpened():
    ret,img=cam.read()
    if not ret:
        break 
    img=np.flip(img,axis=1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower=np.array([0,120,50])
    upper=np.array([10,255,255])
    mask1=cv2.inRange(hsv,lower,upper)
    lower=np.array([170,120,70])
    upper=np.array([180,255,255])
    mask2=cv2.inRange(hsv,lower,upper)

    mask1=mask1+mask2
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2=cv2.bitwise_not(mask1)
    img1=cv2.bitwise_and(img,img,mask=mask2)
    img2=cv2.bitwise_and(bg,bg,mask=mask1)
    finalOutput=cv2.addWeighted(img1,1,img2,1,0)
    outputfile.write(finalOutput)
    cv2.imshow("magic",finalOutput)
    cv2.waitKey(5)
cam.release()
outputfile.release()
cv2.destroyAllWindows()

