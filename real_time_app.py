import sys, time
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
)
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont

from main import classify


class MainWindow(QMainWindow):

    # Setup widgets
    def __init__(self):
        super().__init__()

        self.start = True

        # set window title
        self.setWindowTitle("Accelerometer Classifier - Live")

        # set window dimensions
        self.setGeometry(100, 100, 1000, 800)

        # create button widget for classifying the data
        self.classify_btn = QPushButton("Classify", self)
        self.classify_btn.move(10, 70)
        self.classify_btn.resize(60, 20)
        self.classify_btn.clicked.connect(self.classify_data)

        # create end experiment btn
        self.end_btn = QPushButton("End Experiment", self)
        self.end_btn.move(10, 200)
        self.end_btn.resize(60, 20)
        self.end_btn.clicked.connect(self.end_clicked)

        # create label that will update with Jumping or Walking
        self.label = QLabel("Unknown", self)
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.move(20, 20)

    # fetch request and classify data
    def classify_data(self):

        # As long as we don't press the stop button
        while self.start:
            # read the data from the excel file output from Phyphox
            data = pd.read_excel("http://192.168.0.16/export?format=0")

            # since the xls file will keep getting larger as the expirement continues, only get recent data (last 5000 rows)
            recent_data = data.tail(5000)

            # call the classify function with the recent data
            res = classify(recent_data)
            print(res)

            # if more than half of the labels are jump, display "Jumping"
            if res["label"].value_counts()["jumping"] > (len(res.index) / 2):
                self.label.setText("Jumping")
            else:
                self.label.setText("Walking")

            # to process GUI event changes even in an infinite loop
            QApplication.processEvents()

    def end_clicked(self):
        self.start = False
        sys.exit()


# so we can run it as a script
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # create the MainWindow instance
    main_window = MainWindow()

    # show the main window
    main_window.show()

    # start the GUI event loop
    sys.exit(app.exec_())
