import cv2
import numpy as np
from keras import models
import sys
from PIL import Image
#Load the saved model
model = models.load_model('gestures_trained_cnn_model.h5')
label=['Call','Good','Hi','ILY','Love','Nothing','NotOk','OK','Peace','Thankyou','You']
# Start capturing Video through webcam
video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    kernel = np.ones((3,3),np.uint8)
    cv2.rectangle(frame,(30,230),(230,430),(0,255,0),2)
    aoi=frame[230:430,30:230]
     
    hsv = cv2.cvtColor(aoi, cv2.COLOR_BGR2HSV)
# define range of skin color in HSV
    lower_skin = np.array([0,10,120], dtype=np.uint8)
    upper_skin = np.array([179,120,255], dtype=np.uint8)
#extract skin colur image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#extrapolate the hand to fill dark spots within
    #mask = cv2.dilate(mask,kernel,iterations = 4)
#blur the image
    #mask = cv2.GaussianBlur(mask,(5,5),100)
    mask = cv2.resize(mask,(128,128))
    img_array = np.array(mask)
    #print(img_array.shape)
# Changing dimension from 128x128 to 128x128x3
    img_array = np.stack((img_array,)*3, axis=-1)
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3 
    img_array_ex = np.expand_dims(img_array, axis=0)
    #print(img_array_ex.shape)
    #Calling the predict method on model to predict gesture in the frame
    prediction = model.predict(img_array_ex)
    #print(prediction)
    print(np.take(label,(prediction.argmax(1)-1)))
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()