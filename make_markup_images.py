import cv2
 
# Read the original image
img  = cv2.imread('Synthetic_MICCAI2020_dataset/others/Video_15/ground_truth/000.png')
oimg = cv2.imread('Synthetic_MICCAI2020_dataset/others/Video_15/images/000.png')
 

def create_overlayed_image(original, overlayable):
    edges = cv2.Canny(image=overlayable, threshold1=100, threshold2=200) # Canny Edge Detection
    _, alpha = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    b, g, r = edges, edges, edges
    overlay = cv2.merge([b, g, r, alpha], 4)

    return cv2.add(original, overlay)





# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

 
cv2.imshow('Canny Edge Detection', create_overlayed_image(cv2.cvtColor(oimg, cv2.COLOR_BGR2BGRA), img_blur))
cv2.waitKey(0)
#cv2.imshow('Canny Edge Detection', edges)
#cv2.waitKey(0)
#cv2.imwrite("gfg_white.png", overlay)
 
cv2.destroyAllWindows()