#!/usr/bin/env python
# coding: utf-8

# ### Importing required library

# In[1]:


import cv2
import numpy as np
import dlib


# In[2]:


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


# In[4]:


#Strat the webcam
cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    #Reading Frames
    _, frame = cap.read()
    
    #Smoothing the frame pixels
    frame = cv2.bilateralFilter(frame, 3, 175, 175)
    
    #Detecting faces in webcam
    faces = detector(frame)
    
    #Saving the coordinate of detected face in x1,x2,y1,y2
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #Predicting the face landmark on the detected face
        landmarks = predictor(frame, face)
        l=[]
        #Saving the landmark around the eyes in var l
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            l.append((x,y))
    #Diving the landmarks into left and right eye        
    lefteye=l[0:6]
    righteye=l[6:12]
    lefteye=np.array(lefteye)
    righteye=np.array(righteye)
    
    #Taking convex hull of both eyes
    leftEyeHull = cv2.convexHull(lefteye)
    rightEyeHull = cv2.convexHull(righteye)
    
#     #Creating a mask of same size as the frame
#     mask = np.zeros(frame.shape, dtype=frame.dtype)  

    
    #Calculating the center of left and right eye hull
    M1 = cv2.moments(leftEyeHull)
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])
    M2 = cv2.moments(rightEyeHull)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    
    center1=(cX1,cY1)
    center2=(cX2,cY2)
    
    #Reading the image contaning anime eyes
    src = cv2.imread('eyes.jpg')
    src = cv2.bilateralFilter(src, 3, 175, 175)
    
    #Creating a mask of same size as the src image
    src_mask = np.zeros(src.shape, src.dtype)
    #Specifying the roi of the image for left eye
    poly = np.array([ [12,16], [27,12], [70,15], [81,22], [86,35], [82,41], [62,47],[35,44],[21,33]], np.int32)
    src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    
    
    
    src_mask2 = np.zeros(src.shape, src.dtype)
    
    #Specifying the roi of the image for right eye
    poly2 = np.array([ [156,31], [165,22], [163,23], [191,10], [211,12], [222,14],[230,17] ,[213,40],[190,45],[173,45],[162,42]], np.int32)
    src_mask2 = cv2.fillPoly(src_mask2, [poly2], (255, 255, 255))
    
    #Scale the image to your eye size
    scale_percent = 50 # percent of original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    src = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
    src_mask = cv2.resize(src_mask, dim, interpolation = cv2.INTER_AREA)
    src_mask2 = cv2.resize(src_mask2, dim, interpolation = cv2.INTER_AREA)

    output = cv2.seamlessClone(src,frame , src_mask, center1, cv2.NORMAL_CLONE)

    output=cv2.seamlessClone(src, output, src_mask2, center2, cv2.NORMAL_CLONE)
    
    gamma = 1.8  # change the value here to get different result
    adjusted_output = adjust_gamma(output, gamma=gamma)


    #output the frame
    cv2.imshow('Frame', adjusted_output)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




