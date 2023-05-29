

# from tkinter import *
# from tkinter import ttk
# from PIL import ImageTk, Image
# import numpy as np
# import glob
# import re
# import cv2





# def donothing():
#   print("Action")

# def streamModeAction():
#     if var.get():
#         print("Checkbutton is selected.")
#     else:
#         print("Checkbutton is not selected.")

# # Pathname sorter
# def extract_number(filename):
#     # Extract the numeric part from the filename
#     match = re.search(r'\d+', filename)
#     if match:
#         return int(match.group())
#     return 0

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




# class Picture:
#     def __init__(self, parent):
#         self.parent = parent
#         img = Image.open(file='C:\\workingdir\\orvosirobotika\\HamLyquerModell\\data\\frame0.png')
#         resized_image = img.resize((300,205), Image.ANTIALIAS)
#         new_image= ImageTk.PhotoImage(resized_image)

#         self.label = ttk.Label(self.parent)
#         self.label['image'] = img
#         img.image = new_image
#         self.label.pack()
 
#         btn = Button(self.parent, command=self.cameraStream, text='Test').pack(side='bottom', pady=50)
 
#     def update(self):
#         img = PhotoImage(file='img2.png')
#         self.label['image'] = img
#         img.image = img


#     def cameraStream(self):
#         video_=cv2.VideoCapture(0)
    
#         if not video_.isOpened():
#           print('Faild to open the camera')
#         else:
        
#             while True:
#                 ret, frame = video_.read()

#                 img = ImageTk.PhotoImage(frame)
#                 self.label['image'] = img
#                 img.image = img

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break


# if __name__ == "__main__":
    
        
#     root = Tk()
          
#     var = IntVar()
     
#     root.title("HamLyquerModell")
#     root.minsize(800, 600)
     
#     menubar = Menu(root)
#     root.config(menu = menubar)
     
#     filemenu = Menu(menubar, tearoff=0)
#     filemenu.add_command(label = "Open", command = donothing)
#     menubar.add_cascade(label="File", menu=filemenu)
     
     
#     Picture(root)
    
#     # streamMode = Checkbutton(root, text = "Camera Stream Mode", command = donothing).grid(row=0, sticky=W)
#     # streamMode = Checkbutton(root, text = "Camera Stream Mode", command = donothing)
#     # streamMode.pack()
       
#     # frame = Frame(root)
#     # frame.pack()
#     # frame.place(anchor='center', relx=0.5, rely=0.5)

#     # # Create an object of tkinter ImageTk
#     # img = ImageTk.PhotoImage(Image.open("C:\\workingdir\\orvosirobotika\\HamLyquerModell\\data\\frame0.png"))

#     # # Create a Label Widget to display the text or Image
#     # label = Label(frame, image = img)
#     # label.pack()
    
  
     
     
#     root.mainloop()




import tkinter as tk
import cv2
from PIL import ImageTk, Image
import threading
import multiprocessing 


def donothing():
    print("Action")

class CameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Camera Stream")

        self.video_source = video_source
        self.video_capture = cv2.VideoCapture(self.video_source)
        self.current_frame = None
        
        
        
        self.menu_bar = tk.Menu(window)
        self.window.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open File", command=donothing)#self.play_camera_stream)
        self.menu_bar.entryconfig("File", state = "disable")    
                  
        self.menu_bar.add_command(label="Active Mode: Stream", command=self.changeMode)
        

        self.canvas = tk.Canvas(window, width=self.video_capture.get(3), height=self.video_capture.get(4))
        self.canvas.pack()

        self.start_button = tk.Button(window, text="Start Stream", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(window, text="Stop Stream", command=self.stop_stream)
        self.stop_button.pack(side=tk.LEFT)

        self.is_streaming = False
        self.stream_thread = None        

    def start_stream(self):
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self.update)
            self.stream_thread.start()
          

    def stop_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_thread.join()

    def update(self):
        if self.is_streaming:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = Image.fromarray(self.current_frame)
                self.current_frame = ImageTk.PhotoImage(self.current_frame)
                self.canvas.create_image(0, 0, image=self.current_frame, anchor=tk.NW)
            self.window.after(15, self.update)

    def close_app(self):
        self.stop_stream()
        self.video_capture.release()
        self.window.destroy()

    def changeMode(self):
        if(self.menu_bar.entrycget(2, "label") == "Active Mode: Stream"):
            self.menu_bar.entryconfig("Active Mode: Stream", label = "Active Mode: Video from file")
            self.menu_bar.entryconfig("File", state = "active")    
        else:
            self.menu_bar.entryconfig("Active Mode: Video from file", label = "Active Mode: Stream")
            self.menu_bar.entryconfig("File", state = "disable")    
        print("mode selector")    
        

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == '__main__':
    main()