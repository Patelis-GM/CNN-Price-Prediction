import os
import pandas
from Utils.Curve import Curve


# Utility function to parse a CSV file and create the appropriate list of Curves
def parse(filePath: str, delimiter: str = '\t'):

    if os.stat(filePath).st_size == 0 or not filePath.lower().endswith(".csv"):
        print("Error : File {} is either empty or does not have the .csv extension".format(filePath))
        return []

    dataFrame = pandas.read_csv(filePath, delimiter, header=None)
    dataSet = dataFrame.values
    ids = dataSet[:, 0]
    dataSet = dataSet[:, 1:]
    dataSet = dataSet.astype('float32')
    curves = []
    for i in range(0, dataSet.shape[0]):
        curve = Curve(ids[i], dataSet[i, :])
        curves.append(curve)
    return curves
