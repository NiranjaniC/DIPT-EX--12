# DIPT-EX-12 Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program

```
Developed by :  Niranjani.C
Registration Number :  212223220069
```
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
with_glass = cv2.imread('with_glass.png', 0)
with_out_glass = cv2.imread('face.jpg', 0)
group_photo = cv2.imread('group.png', 0)
plt.imshow(with_glass,cmap='gray')
plt.imshow(with_out_glass,cmap='gray')
plt.imshow(group_photo,cmap='gray')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    
    face_img = img.copy()
    
    faces = face_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), 255, 10)
        
    return face_img
result = detect_face(with_glass)
plt.imshow(result,cmap='gray')
result = detect_face(with_out_glass)
plt.imshow(result,cmap='gray')
# Gets errors!
result = detect_face(group_photo)
plt.imshow(result,cmap='gray')
def adj_detect_face(img):
    
    face_img = img.copy()
    
    faces = face_cascade.detectMultiScale(
        face_img,
        scaleFactor=1.2,
        minNeighbors=5
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), 255, 10)
        
    return face_img
# Doesn't detect the side face.
result = adj_detect_face(group_photo)
plt.imshow(result,cmap='gray')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_eyes(img):
    
    face_img = img.copy()
    
    eyes = eye_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), 255, 10)
        
    return face_img
result = detect_eyes(with_out_glass)
plt.imshow(result, cmap='gray')
plt.axis('off')
eyes = eye_cascade.detectMultiScale(group_photo)
# White around the pupils is not distinct enough to detect eyes here!
result = detect_eyes(group_photo)
plt.imshow(result,cmap='gray')
cap = cv2.VideoCapture(0) 

while True: 
    
    ret, frame = cap.read(0) 
     
    frame = detect_face(frame)
 
    cv2.imshow('Video Face Detection', frame) 
 
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
        
cap.release() 
cv2.destroyAllWindows()
```

## Output

<img width="630" height="617" alt="image" src="https://github.com/user-attachments/assets/974fb604-31ca-40d4-b7bd-9297fdfbb8d7" />
<img width="518" height="412" alt="image" src="https://github.com/user-attachments/assets/a15f36a1-811f-4f64-90f3-ac0d703704f8" />
<img width="721" height="417" alt="image" src="https://github.com/user-attachments/assets/5d2d38e8-354d-4827-821b-340fe8fdffec" />
<img width="470" height="422" alt="image" src="https://github.com/user-attachments/assets/c60d2ac1-7c48-49e0-bff3-6dbf685bb32d" />
<img width="545" height="420" alt="image" src="https://github.com/user-attachments/assets/31511631-bb43-4e55-9a8c-bf7256461434" />
<img width="676" height="426" alt="image" src="https://github.com/user-attachments/assets/43b58466-14c4-42f1-b919-e01188be09e4" />
<img width="736" height="368" alt="image" src="https://github.com/user-attachments/assets/a9125b9d-a380-4b15-bf96-929c438a10f8" />
<img width="617" height="395" alt="image" src="https://github.com/user-attachments/assets/2c4e462f-7d82-429d-b142-77bd922e9b7a" />
<img width="801" height="627" alt="Screenshot 2026-09-05 153618" src="https://github.com/user-attachments/assets/64a8a3c2-c035-4f16-a6fc-5d53086fe456"/>

## Result
Thus, Face Detection using Haar Cascades using OpenCV and Matplotlib is executed successsfully.






