from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QApplication, QLabel, QPushButton,
    QFileDialog, QSlider, QMessageBox
)
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
import cv2
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)  # Signal for 2 video frames
    update_slider_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap1 = None
        self.cap2 = None
        self.frame_count1 = 0
        self.frame_count2 = 0
        self.current_frame1 = 0
        self.current_frame2 = 0

    def run(self):
        while self._run_flag:
            if self.cap1 and self.cap1.isOpened() and self.cap2 and self.cap2.isOpened():
                ret1, cv_img1 = self.cap1.read()
                ret2, cv_img2 = self.cap2.read()
                if ret1 and ret2:
                    self.current_frame1 += 1
                    self.current_frame2 += 1
                    self.change_pixmap_signal.emit(cv_img1, cv_img2)
                    self.update_slider_signal.emit(
                        int((self.current_frame1 / self.frame_count1) * 100)
                    )
                QThread.msleep(33)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        self.wait()

    def open_video_folder(self, folder_path):
        """Opens the video folder and finds std_view.mp4 and bev_view.mp4"""
        self.cap1 = cv2.VideoCapture(os.path.join(folder_path, "std_view.mp4"))
        self.cap2 = cv2.VideoCapture(os.path.join(folder_path, "bev_view.mp4"))
        self.frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame1 = 0
        self.current_frame2 = 0

    def play_video(self):
        """Sets run flag to True and starts the thread"""
        self._run_flag = True
        self.start()

    def pause_video(self):
        """Sets run flag to False to pause the thread"""
        self._run_flag = False


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tennis Analysis")
        self.setFixedSize(1920, 1080)

        self.disply_width = 1280
        self.display_height = 720

        # create the label that holds the image
        self.video_label = QLabel(self)
        self.video_label.setGeometry(
            QtCore.QRect(10, 10, 1280, 720)
        )  # Set the label position and size
        self.video_label.setFrameShape(
            QtWidgets.QFrame.Box
        )  # Add a box around the label

        # create the label that holds the image
        self.minimap_label = QLabel(self)
        self.minimap_label.setGeometry(
            QtCore.QRect(1350, 10, 360, 640)
        )  # Set the label position and size
        self.minimap_label.setFrameShape(QtWidgets.QFrame.Box)

        # create buttons
        self.open_button = QPushButton("Open Folder", self)
        self.open_button.setGeometry(QtCore.QRect(10, 750, 89, 25))
        self.open_button.clicked.connect(self.open_video_dialog)

        self.play_button = QPushButton("Play", self)
        self.play_button.setGeometry(QtCore.QRect(109, 750, 89, 25))
        self.play_button.clicked.connect(self.play_video)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(QtCore.QRect(208, 750, 89, 25))
        self.stop_button.clicked.connect(self.stop_video)

        # create a slider for video progress
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setGeometry(QtCore.QRect(307, 753, self.disply_width - 295, 20))  # Set position and size

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_slider_signal.connect(self.update_slider)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def open_video_dialog(self):
        """Opens a file dialog to select a folder containing two video files with predefined names"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Videos",
            "",
            options=options,
        )
        if folder:
            # Define the expected video file names
            video1 = os.path.join(folder, "std_view.mp4")
            video2 = os.path.join(folder, "bev_view.mp4")
            
            # Check if both video files exist
            if os.path.exists(video1) and os.path.exists(video2):
                # Open the video files
                self.thread.open_video_folder(folder)
                self.slider.setValue(0)
            else:
                QMessageBox.warning(self, "Error", "The selected folder does not contain both std_view.mp4 and bev_view.mp4")

    def play_video(self):
        """Play the video"""
        self.thread.play_video()

    def stop_video(self):
        """Stop the video"""
        self.thread.pause_video()
        # Get the current frame position
        current_frame = self.thread.current_frame1  # Assuming we use the frame count from the first video
        # Convert frame number to percentage
        slider_value = int((current_frame / self.thread.frame_count1) * 100)
        self.slider.setValue(slider_value)

    @pyqtSlot(int)
    def update_slider(self, value):
        """Update the slider position"""
        self.slider.setValue(value)

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_image(self, cv_img1, cv_img2):
        """Updates the image_label with new opencv images"""
        qt_img1 = self.convert_cv_qt(cv_img1)
        self.video_label.setPixmap(qt_img1)
        qt_img2 = self.convert_cv_qt(cv_img2, self.minimap_label.width(), self.minimap_label.height())
        self.minimap_label.setPixmap(qt_img2)

    def convert_cv_qt(self, cv_img, width=None, height=None):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        if width and height:
            p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        else:
            p = convert_to_Qt_format.scaled(
                self.disply_width, self.display_height, Qt.KeepAspectRatio
            )
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
