import cv2
import numpy as np

OVERLAY_COLOR = [128,0,128,255]
OUTLINE_WIDTH = (5, 5)

def show(img):
    cv2.imshow('Img', img)
    cv2.waitKey(0)

def img_to_4(img):
    b, g, r = cv2.split(img)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    return cv2.merge([b, g, r, alpha], 4)

def create_overlayed_image(original, overlayable):
    original    = img_to_4(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    overlayable = cv2.cvtColor(overlayable, cv2.COLOR_BGR2RGB)
    overlayable4 = img_to_4(cv2.cvtColor(overlayable, cv2.COLOR_BGR2RGB))

    edges = cv2.Canny(image=overlayable, threshold1=100, threshold2=200) # Canny Edge Detection
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OUTLINE_WIDTH))
    _, alpha = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    b, g, r = edges, edges, edges
    edges = cv2.merge([b, g, r, alpha], 4)

    edges[np.where((edges==[255,255,255,255]).all(axis=2))] = OVERLAY_COLOR  # recolor to purple
    overlayable4[np.where((overlayable4==[255,255,255,255]).all(axis=2))] = OVERLAY_COLOR
    overlay = cv2.addWeighted(overlayable4, .3, edges, .7, 0)

    return cv2.cvtColor(cv2.cvtColor(cv2.add(original, overlay), cv2.COLOR_BGRA2RGB), cv2.COLOR_RGB2BGR)


def sample():
    # Read the original image
    img  = cv2.imread('Synthetic_MICCAI2020_dataset/others/Video_15/ground_truth/000.png')
    oimg = cv2.imread('Synthetic_MICCAI2020_dataset/others/Video_15/images/000.png')

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    cv2.imshow('Canny Edge Detection', create_overlayed_image(oimg, img_blur))
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


#sample()
