import random
import sys
import numpy
from matplotlib import pyplot

from Utils.Curve import Curve
from Utils.Parser import parse
from tensorflow.keras.models import load_model
from Utils.ArgumentParser import ArgumentParser

if __name__ == '__main__':

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)
    argumentParser.addArgument(argument="-q", type="path", mandatory=True)
    argumentParser.addArgument(argument="-od", type="path", mandatory=True)
    argumentParser.addArgument(argument="-oq", type="path", mandatory=True)
    argumentParser.addNumericArgument(argument="-n", type="int", floor=1, ceiling=359, mandatory=False)

    if not argumentParser.parse(sys.argv):
        exit(1)

    datasetPath = argumentParser.getArgument("-d")
    queryPath = argumentParser.getArgument("-q")
    outputDatasetPath = argumentParser.getArgument("-od")
    outputQueryPath = argumentParser.getArgument("-oq")
    n = argumentParser.getArgument("-n")

    if n is None:
        n = 0

    curves = parse(datasetPath)

    if len(curves) == 0:
        exit(1)


    if datasetPath != queryPath:
        print("Error: The '-d' parameter should match the '-q' parameter")
        exit(1)

    if outputDatasetPath == outputQueryPath:
        print("Error: The '-od' parameter should not match the '-oq' parameter")
        exit(1)

    encoder = load_model("Model")

    # Keep the Encoder part of the Autoencoder
    for i in range(4):
        encoder.pop()

    # Split each Curve into windows
    #
    # Given :
    # - Curve C with values V = [1,2,3,4,5,6]
    # - Window = 3
    #
    # Windowed values of C are :
    # - W = [ [ [1] , [2] , [3] ] , [ [4] , [5] , [6] ] ]
    # - where the shape of W is ((length(C) / Window) , Window , 1)
    window = 10
    normalisedWindowedCurves = []
    for curve in curves:
        normalisedValues = curve.normalise(curve.getValues())
        normalisedWindowedValues = numpy.reshape(normalisedValues, (-1, window, 1))
        normalisedWindowedCurves.append(Curve(curve.getID(), normalisedWindowedValues))

    # Compress each Curve using the Encoder part of the Autoencoder
    compressedCurves = []
    for i in range(len(normalisedWindowedCurves)):
        normalisedWindowedValues = normalisedWindowedCurves[i].getValues()
        compressed = encoder.predict(normalisedWindowedValues)
        compressed = numpy.reshape(compressed, (-1))
        compressed = curves[i].denormalise(compressed)
        compressedCurves.append(Curve(curves[i].getID(), compressed))

    # Output the first 350 compressed Curves into the Output-Dataset file
    with open(outputDatasetPath, 'w') as outputFile:
        for i in range(0, 350):
            outputFile.write(compressedCurves[i].toCSV())
            outputFile.write('\n')
        outputFile.close()

    # Output the last 9 compressed Curves into the Output-Query file
    with open(outputQueryPath, 'w') as outputFile:
        for i in range(350, 359):
            outputFile.write(compressedCurves[i].toCSV())
            outputFile.write('\n')
        outputFile.close()

    if n > 0:

        # Select at random the indices of the Curves that will be plotted alongside their compressed counterpart
        indices = list(range(len(curves)))
        random.shuffle(indices)
        indices = indices[:n]

        for index in indices:
            curve = curves[index]
            compressedCurve = compressedCurves[index]

            figure, axes = pyplot.subplots(2)

            axes[0].plot(curve.getValues(), label=curve.getID(), color='blue')
            axes[0].legend()

            axes[1].plot(compressedCurve.getValues(), label="Compressed " + curve.getID(), color='red')
            axes[1].legend()

            pyplot.show()
