import typing
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
from lib.util import video_to_frame
import numpy as np

class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list,int,int])   
    _run_flag=True
    def __init__(self,video_path):
        super(VideoToImagesThread, self).__init__()
        self.video_path=video_path
    def run(self):
        # capture from web cam
        video_images, fps, count=video_to_frame(self.video_path)
        _run_flag=False
        self.emit_signal.emit(video_images, fps, count)
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video to image thread")
    
    def isFinished(self):
        print(" finish thread")
           