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
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import shutil


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # set window title
        self.setWindowTitle("Accelerometer Classifier")

        # set window dimensions
        self.setGeometry(100, 100, 600, 400)

        # create label widget for displaying instructions
        self.instructions_label = QLabel("Select the input CSV file", self)
        self.instructions_label.move(10, 10)
        self.instructions_label.resize(300, 20)

        # create text edit widget for displaying the input file path
        self.input_file_edit = QTextEdit("", self)
        self.input_file_edit.move(10, 40)
        self.input_file_edit.resize(300, 20)
        self.input_file_edit.setReadOnly(True)

        # create button widget for selecting the input file
        self.input_file_button = QPushButton("Select", self)
        self.input_file_button.move(320, 40)
        self.input_file_button.resize(60, 20)
        self.input_file_button.clicked.connect(self.select_input_file)

        # create button widget for classifying the data
        self.classify_button = QPushButton("Classify", self)
        self.classify_button.move(10, 70)
        self.classify_button.resize(60, 20)
        self.classify_button.clicked.connect(self.classify_data)

        # create text edit widget for displaying the output file path
        self.output_file_edit = QTextEdit("", self)
        self.output_file_edit.move(10, 100)
        self.output_file_edit.resize(300, 20)
        self.output_file_edit.setReadOnly(True)

        # create button widget for saving the output file
        self.save_button = QPushButton("Save", self)
        self.save_button.move(320, 100)
        self.save_button.resize(60, 20)
        self.save_button.clicked.connect(self.save_output_file)

        # create label widget for displaying the plot
        self.plot_label = QLabel("", self)
        self.plot_label.move(10, 130)
        self.plot_label.resize(580, 250)

    def select_input_file(self):
        # open file dialog to select the input file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select input file", "", "CSV Files (*.csv)", options=options
        )

        # update the input file path text edit widget
        self.input_file_edit.setPlainText(file_path)

    def classify_data(self):
        # read the input file using pandas
        input_file_path = self.input_file_edit.toPlainText()
        data = pd.read_csv(input_file_path)

        # classify the data here using your classifier
        # ...
        # replace the 'label' column with the classified labels

        # add a new column with dummy labels
        data["label"] = ["walking" if x < 1000 else "jumping" for x in range(len(data))]

        # save the updated CSV file
        # data.to_csv("input_with_labels.csv", index=False)

        # save the output file as CSV
        output_file_path = input_file_path.replace(".csv", "_output.csv")
        data.to_csv(output_file_path, index=False)

        # update the output file path text edit widget
        self.output_file_edit.setPlainText(output_file_path)

        # generate the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            data["Time (s)"],
            data["Linear Acceleration y (m/s^2)"],
            label="Accelerometer Data",
        )
        ax.scatter(
            data["Time (s)"],
            data["Linear Acceleration y (m/s^2)"],
            c=data["label"].apply(lambda x: "r" if x == "jumping" else "b"),
            label="Classified Labels",
        )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.set_title("Accelerometer Data Classification")
        ax.legend()
        fig.canvas.draw()

        # convert the plot to a QImage
        plot_image = fig.canvas.buffer_rgba()
        plot_qimage = QImage(
            plot_image, plot_image.shape[1], plot_image.shape[0], QImage.Format_RGBA8888
        )

        # set the plot image to the label widget
        self.plot_label.setPixmap(QPixmap.fromImage(plot_qimage))

    def save_output_file(self):
        # open file dialog to select the output file path
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save output file", "", "CSV Files (*.csv)", options=options
        )

        # save the output file to the selected path
        output_file_path = self.output_file_edit.toPlainText()
        if file_path:
            shutil.copy(output_file_path, file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # create the MainWindow instance
    main_window = MainWindow()

    # show the main window
    main_window.show()

    # start the GUI event loop
    sys.exit(app.exec_())
