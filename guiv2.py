

import tkinter as tk
import cv2
from PIL import ImageTk, Image
import threading
import videoeditor as ve
from tkinter import Tk     
from tkinter.filedialog import askopenfilename
import model as md
import make_markup_images as mp
import numpy as np


def donothing():
    print("Action")

class CameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("HamLycurelModell")
        
        self.model = md.create_model()

        if md.model_exists():
            print("Load Model Weights")
            self.model.load_weights(md.WEIGHTS_PATH)

        self.filePath = ""
        self.mode = "STREAM"
        self.video_source = video_source
        self.video_capture = cv2.VideoCapture(self.video_source)
        self.video_capture_from_file = cv2.VideoCapture("video.mp4")
        self.current_frame = None       
              
        
        self.menu_bar = tk.Menu(window)
        self.window.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open File", command=self.openFileBrowser)
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
        if("STREAM" == self.mode):
            if not self.is_streaming:
                self.is_streaming = True
                self.stream_thread = threading.Thread(target=self.update)
                self.stream_thread.start()
        else:
            if(len(self.filePath) == 0):
                print("There is no active file selected!!")
            else:
                 if not self.is_streaming:
                    ve.videoSlicer(self.filePath, self.model)               
                    self.is_streaming = True
                    self.video_capture_from_file = cv2.VideoCapture("video.mp4")
                    self.stream_thread = threading.Thread(target=self.update)
                    self.stream_thread.start() 

    def stop_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            # self.stream_thread.join()
        if self.mode != "STREAM":
            self.video_capture_from_file = cv2.VideoCapture("video.mp4")
            self.is_streaming = True
            self.stop_button["state"] = tk.DISABLED
            self.window.after(15, self.update)

    def update(self):
        if(self.menu_bar.entrycget(2, "label") == "Active Mode: Stream"):
            if self.is_streaming:
                ret, frame = self.video_capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    original_img = md.load_images_camera(frame)
                    preds_test  = self.model.predict(original_img)
                    overlayable = np.squeeze(((preds_test[0] > .5) * 255).astype(np.uint8))    
                    modified_image = mp.create_overlayed_image(original_img[0], overlayable)
                    
                    self.current_frame = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
                    self.current_frame = Image.fromarray(self.current_frame)
                    self.current_frame = ImageTk.PhotoImage(self.current_frame)                        
                    self.canvas.create_image(0, 0, image=self.current_frame, anchor=tk.NW)
                    self.window.after(15, self.update)
        else:            
             if self.is_streaming:
                ret, frame = self.video_capture_from_file.read()
                # print(ret)
                if ret:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #COLOR_BGR2RGB
                    self.current_frame = Image.fromarray(self.current_frame)
                    self.current_frame = ImageTk.PhotoImage(self.current_frame)
                    self.canvas.create_image(0, 0, image=self.current_frame, anchor=tk.NW)
                    self.window.after(15, self.update)
                else:
                    self.video_capture_from_file.release()
                    self.is_streaming = False
                    self.stop_button["state"] = tk.ACTIVE
                    print("Thread end")
                                        
                    # self.stream_thread.join()

    def close_app(self):
        self.stop_stream()
        self.video_capture.release()
        self.window.destroy()

    def changeMode(self):
        if(self.menu_bar.entrycget(2, "label") == "Active Mode: Stream"):
            self.menu_bar.entryconfig("Active Mode: Stream", label = "Active Mode: Video from file")
            self.menu_bar.entryconfig("File", state = "active")
            self.start_button["text"] = "Start Conversion"
            self.stop_button["text"] = "Play again"
            self.stop_button["state"] = tk.DISABLED
            self.mode = "CONVERSION"    
        else:
            self.menu_bar.entryconfig("Active Mode: Video from file", label = "Active Mode: Stream")
            self.menu_bar.entryconfig("File", state = "disable")
            self.start_button["text"] = "Start Stream"
            self.stop_button["text"] = "Stop Stream"
            self.mode = "STREAM" 
            self.stop_button["state"] = tk.ACTIVE    
        print("mode selector") 
                
    def openFileBrowser(self):
        Tk().withdraw()
        self.filePath = askopenfilename()
        # Throw an error message if the file extension isn't correct.
        print(self.filePath)   
        

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == '__main__':
    main()