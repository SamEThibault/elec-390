import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QLabel,
    QPushButton,
    QTextEdit,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib.pyplot as plt

from main import classify, train


class MainWindow(QMainWindow):

    # Setup widgets
    def __init__(self):
        super().__init__()

        # start by training the model so we can use it to classify
        self.model = train()

        self.start = True

        # set window title
        self.setWindowTitle("Accelerometer Classifier")

        # set window dimensions
        self.setGeometry(100, 100, 1000, 800)

        # create label widget for input
        self.input_label = QLabel("Select the input CSV file", self)
        self.input_label.move(10, 10)
        self.input_label.resize(300, 20)

        # create text edit widget for displaying the input file path
        self.input_file_edit = QTextEdit("", self)
        self.input_file_edit.move(10, 40)
        self.input_file_edit.resize(300, 20)
        self.input_file_edit.setReadOnly(True)

        # create button widget for selecting the input file
        self.input_file_btn = QPushButton("Select", self)
        self.input_file_btn.move(320, 40)
        self.input_file_btn.resize(60, 20)
        self.input_file_btn.clicked.connect(self.select_input_file)

        # create button widget for classifying the data
        self.classify_btn = QPushButton("Classify", self)
        self.classify_btn.move(10, 70)
        self.classify_btn.resize(60, 20)
        self.classify_btn.clicked.connect(self.classify_data)

        # create text edit widget for displaying the output file path
        self.output_file_edit = QTextEdit("", self)
        self.output_file_edit.move(10, 100)
        self.output_file_edit.resize(300, 20)
        self.output_file_edit.setReadOnly(True)

        # create label widget for displaying the plot
        self.plot_label = QLabel("", self)
        self.plot_label.move(10, 130)
        self.plot_label.resize(1000, 600)

        # ----------- LIVE -------------
        # create button widget for classifying the data LIVE
        self.classify_live_btn = QPushButton("Classify in real time", self)
        self.classify_live_btn.move(480, 70)
        self.classify_live_btn.resize(130, 20)
        self.classify_live_btn.clicked.connect(self.classify_data_live)

        # create end experiment btn
        self.end_btn = QPushButton("End Experiment", self)
        self.end_btn.move(480, 125)
        self.end_btn.resize(125, 20)
        self.end_btn.clicked.connect(self.end_clicked)

        # create label that will update with Jumping or Walking
        self.label = QLabel("Live Experiment Not Started", self)
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.resize(300, 30)
        self.label.move(480, 20)

    # get input file from user
    def select_input_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select input file", "", "CSV Files (*.csv)", options=options
        )

        # update the input file path text edit widget
        self.input_file_edit.setPlainText(file_path)

    # call the classifier and plot the results
    def classify_data(self):

        # read the input file
        input_file_path = self.input_file_edit.toPlainText()
        data = pd.read_csv(input_file_path)

        # from main.py
        res = classify(data, self.model)
        data = data[: len(res)]

        # save the output file as CSV into the app_output folder
        new_file_name = input_file_path.replace(".csv", "_output.csv").split("/")[-1]
        output_file_path = "./app_output/" + new_file_name
        res.to_csv(output_file_path, index=False)

        # update the output file path text edit widget
        self.output_file_edit.setPlainText(output_file_path)

        # generate the plot
        # 9x6 inches figure
        fig, ax = plt.subplots(figsize=(9, 6))

        # plots scatter plot of time vs acceleration where red dots are walking classifications, jumping data is blue
        ax.scatter(
            data["Time (s)"],
            data["Absolute acceleration (m/s^2)"],
            c=res["label"].apply(lambda x: "r" if x == "jumping" else "b"),
            s=2,
            label="Jumping",
        )

        ax.scatter(
            data["Time (s)"],
            data["Absolute acceleration (m/s^2)"],
            c=res["label"].apply(lambda x: "b" if x == "walking" else "r"),
            s=2,
            label="Walking",
        )

        ax.legend()
        legend = ax.get_legend()
        legend.legendHandles[0].set_color("blue")
        legend.legendHandles[1].set_color("red")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absolute Acceleration (m/s^2)")
        ax.set_title("Accelerometer Data Classification")
        fig.canvas.draw()

        # convert the plot to a QImage
        plot_image = fig.canvas.buffer_rgba()
        plot_qimage = QImage(
            plot_image, plot_image.shape[1], plot_image.shape[0], QImage.Format_RGBA8888
        )

        # set the plot image to the label widget
        self.plot_label.setPixmap(QPixmap.fromImage(plot_qimage))

    # fetch request and classify data
    def classify_data_live(self):
        self.start = True

        # As long as we don't press the stop button
        while self.start:
            # read the data from the excel file output from Phyphox
            try:
                data = pd.read_excel("http://192.168.0.16/export?format=0")
            except:
                break

            # since the xls file will keep getting larger as the expirement continues, only get recent data (last 5000 rows)
            recent_data = data.tail(1000)

            # call the classify function with the recent data
            res = classify(recent_data, self.model)

            # if at least 50% of the labels say jumping, then display jumping
            if res["label"].value_counts()[0] > 400:
                self.label.setText("Jumping")
            else:
                self.label.setText("Walking")

            # to process GUI event changes even in an infinite loop
            QApplication.processEvents()

    def end_clicked(self):
        self.start = False


# so we can run it as a script
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # create the MainWindow instance
    main_window = MainWindow()

    # show the main window
    main_window.show()

    # start the GUI event loop
    sys.exit(app.exec_())
