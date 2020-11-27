import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
model=tf.keras.models.load_model('minorprojekt.h5')
a = np.zeros([300,300],dtype='uint8')

print("Press p for predictation")
print("press c for clear")
print("press Esc for quit")
wname='Digits'
drawing='False'
cv2.namedWindow(wname)
def digits(event,x,y,flags,param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
    elif (event == cv2.EVENT_MOUSEMOVE):
        if (drawing==True):
            cv2.rectangle(a,(x,y),(x+10,y+10),(255,255,255),-5)
    elif event== cv2.EVENT_LBUTTONUP:
        drawing=False
cv2.setMouseCallback(wname,digits)
while True:
    cv2.imshow(wname,a)
    key=cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("p"):
        digit = a[:,:]
        digit= cv2.resize(digit,(28,28)).reshape(-1,784)
        print(np.argmax(model.predict(digit)))
    elif key == ord("c"):
        a[:,:]=0
        
cv2.destroyAllWindows()


