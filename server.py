import time
import pandas as pd
from main import classify

while True:
    # Read the data from the excel file output from Phyphox
    data = pd.read_excel("http://192.168.0.16/export?format=0")

    # Since the xls file will keep getting larger as the expirement continues, only get recent data (last 5000 rows)
    recent_data = data.tail(5000)

    # Call the classify function with the recent data
    print(classify(recent_data))

    # Sleep for 3 seconds before fetching the next batch of data
    # time.sleep(3)
