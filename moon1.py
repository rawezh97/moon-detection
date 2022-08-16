import cv2
import numpy

video = cv2.VideoCapture("moon.mp4")
object_detection = cv2.createBackgroundSubtractorMOG2()
while True:
    boolean , frame = video.read()
    frame = cv2.resize(frame,(700,700))
    copyframe = frame.copy()

    imgray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    imblur = cv2.GaussianBlur(imgray,(5,5),1)
    imcanny = cv2.Canny(imblur,150,0)
    
    countours , hieararchy = cv2.findContours(imcanny , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(copyframe , countours , -1 , (0,255,0),2)
  
    cv2.imshow("copyframe",copyframe)
    cv2.imshow("gray",imgray)
    #cv2.imshow("blur",imblur)
    cv2.imshow("canny",imcanny)
    #cv2.imshow("frame",frame)
    cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.distroyAllWindows()
