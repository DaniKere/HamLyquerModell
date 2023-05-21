import cv2

# Load the pre-trained Haar Cascade classifier for boar detection
boar_cascade = cv2.CascadeClassifier('hogcascade_pedestrians.xml')

# Load the image to be processed
img = cv2.imread('boar_image.jpg')
cv2.imshow('vmi',img)
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect boars in the image using the 
boars = boar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw bounding boxes around the detected boars
for (x, y, w, h) in boars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the image with the detected boars
cv2.imshow('Boar Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
