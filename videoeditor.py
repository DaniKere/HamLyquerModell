
# Importing all necessary libraries
import cv2
import os
import numpy as np
import glob
import re

pathVideo = "dummyvideo"
dummyVideo = pathVideo + "\\mouse.mp4"
pathImages = "data\\"

# # Create images from video
# def videoSlicer(dummyVideo):
#    # Read the video from specified path
#     cam = cv2.VideoCapture(dummyVideo)
  
#     try:
#     # creating a folder named data
#         if not os.path.exists('data'):
#             os.makedirs('data')
  
#     # if not created then raise error
#     except OSError:
#         print ('Error: Creating directory of data')
  
#     # frame
#     currentframe = 0
  
#     while(True):
      
#         # reading from frame
#         ret,frame = cam.read()
  
#         if ret:
#             # if video is still left continue creating images
#             name = './data/frame' + str(currentframe) + '.png'
#             print ('Creating...' + name)
  
#             # writing the extracted images
#             cv2.imwrite(name, frame)
  
#             # increasing counter so that it will show how many frames are created            
#             currentframe += 1
#         else:
#             break
  
#     # Release all space and windows once done
#     cam.release()
#     cv2.destroyAllWindows() 

# # Pathname sorter
# def extract_number(filename):
#     # Extract the numeric part from the filename
#     match = re.search(r'\d+', filename)
#     if match:
#         return int(match.group())
#     return 0

# # Create vide from frames
# def videBuilder(pathImages):
#     img_array = []
#     filenames = glob.glob(pathImages + '*.png')
#     # Sort the filenames
#     sorted_filenames = sorted(filenames, key=extract_number)

#     for filename in sorted_filenames:
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width,height)
#         img_array.append(img)
   
#     out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

#     for i in range(len(img_array)):
#         out.write(img_array[i])
#     out.release()


# Create vide from frames
def videBuilder(imgArr):
   
    height, width, layers =  imgArr[0].shape
    size = (width,height)
      
    out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, size)

    for i in range(len(imgArr)):
        out.write(imgArr[i])
    out.release()
    
        
    # Create images from video
def videoSlicer(videoInputPath):
   # Read the video from specified path
    img_array = []
    cam = cv2.VideoCapture(videoInputPath)
  
    try:
    # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
  
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
  
    # frame
    currentframe = 0
  
    while(True):
      
        # reading from frame
        ret,frame = cam.read()
  
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.png'
            print ('Creating...' + name)
  
            # writing the extracted images
            # cv2.imwrite(name, frame)
            img_array.append(frame)
  
            # increasing counter so that it will show how many frames are created            
            currentframe += 1
        else:
            break
  
    # Release all space and windows once done
    videBuilder(img_array)
    cam.release()
    cv2.destroyAllWindows() 
        
# videoSlicer(dummyVideo)
