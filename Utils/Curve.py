import random
from typing import Union
import numpy


class Curve:
    # Static variable of the Curve class that is used in Curve.normalise/denormalise
    __divisor = 10.0

    def __init__(self, id: Union[int, str, float], values: numpy.ndarray):
        self.__id = str(id)
        self.__values = numpy.array(values)
        self.__min = numpy.min(self.__values)
        self.__max = numpy.max(self.__values)

    # Implementation of the __len__() function in the context of the Curve class
    def __len__(self):
        return len(self.__values)

    # Utility function to sample a Curve as in the following example
    #
    # Given :
    # - A Curve C with the following values V = [1,2,3,4,5,6,7,8,9,10]
    # - The function call X,Y = C.sample(timesteps=4,length=8,front=True,includeY=True,normalise=False)
    #
    # X will be [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]] and Y will be [[5],[6],[7],[8]]
    def sample(self, timesteps: int, length: int, front: bool, includeY: bool, normalise: bool, a: float = 0.0, b: float = 1.0):
        xSample = []
        ySample = []
        values = None

        if length <= 0 or length > len(self):
            length = len(self)

        if front:
            values = self.__values[:length]
        else:
            values = self.__values[len(self) - length:]

        if normalise and a < b:
            values = self.normalise(values, a, b)

        for i in range(timesteps, len(values)):
            xSample.append(values[(i - timesteps):i])
            if includeY:
                ySample.append([values[i]])

        if includeY:
            return numpy.array(xSample), numpy.array(ySample)
        else:
            return numpy.array(xSample)

    # Utility static function to sample a set of Curves as in the following example
    #
    # Given :
    # - Curve C1 with the following values V1 = [1,2,3,4,5]
    # - Curve C2 with the following values V2 = [10,20,30,40,50]
    # - The function call Curve.sampleSet([C1,C2],timesteps=3,length=4,front=True,includeY=True,normalise=False)
    #
    # X will be [[1,2,3],[2,3,4],[10,20,30],[20,30,40]] and Y will be [[4],[5],[40],[50]]
    @staticmethod
    def sampleSet(dataset: list, timesteps: int, length: int, front: bool, includeY: bool, normalise: bool, a: float = 0.0, b: float = 1.0):

        xSample = []
        ySample = []

        if length <= 0 or length > len(dataset[0]):
            length = len(dataset[0])

        for curve in dataset:

            values = curve.getValues()

            if normalise and a < b:
                values = curve.normalise(values, a, b)

            if front:
                values = values[:length]

            else:
                values = values[len(curve) - length:]

            for i in range(timesteps, len(values)):
                xSample.append(values[(i - timesteps):i])
                if includeY:
                    ySample.append([values[i]])

        if includeY:
            return numpy.array(xSample), numpy.array(ySample)

        return numpy.array(xSample)

    # Utility function to normalise a sequence in the [a,b] range using the following linear transformation :
    # Normalised-Sequence = < (b-a) * [{Sequence - CurveObject.min}/{CurveObject.max - CurveObject.min}] > + a
    def normalise(self, sequence: numpy.ndarray, a: float = 0.0, b: float = 1.0):
        if a < b:
            if self.__max > self.__min:
                sequence = ((b - a) * ((sequence - self.__min) / (self.__max - self.__min))) + a
            else:
                sequence = (sequence / self.__max)
                sequence = sequence / Curve.__divisor

        return sequence

    # Utility function to denormalise a sequence that was normalised in the [a,b] range using the following linear transformation :
    # Denormalised-Sequence = < (Sequence - a) * [{CurveObject.max - CurveObject.min}/{b - a}] > + CurveObject.min
    def denormalise(self, sequence: numpy.ndarray, a: float = 0.0, b: float = 1.0):
        if a < b:
            if self.__max > self.__min:
                sequence = (((sequence - a) * (self.__max - self.__min)) / (b - a)) + self.__min
            else:
                sequence = sequence * Curve.__divisor
                sequence = sequence * self.__max

        return sequence

    # Utility static function to split a set of Curves into 2 seperate sets
    @staticmethod
    def splitSet(dataset: list, value: Union[int, float], asPercentage: bool = False, shuffle: bool = True):
        if shuffle:
            random.shuffle(dataset)

        if asPercentage:
            if value <= 0 or value > 1:
                value = 0.5
            setOneSize = int(value * len(dataset))
            return dataset[:setOneSize], dataset[setOneSize:]
        else:
            if int(value) <= 0 or int(value) > len(dataset):
                value = int(len(dataset) / 3)
            setOneSize = value
            return dataset[:setOneSize], dataset[setOneSize:]

    def getValues(self):
        return self.__values

    def getID(self):
        return self.__id

    # Utility function to create the CSV representation of a Curve object
    def toCSV(self, delimiter: str = "\t"):
        csvRepresenation = self.__id + delimiter
        for i in range(len(self.__values)):
            csvRepresenation += str(self.__values[i])
            if i != (len(self.__values) - 1):
                csvRepresenation += delimiter
        return csvRepresenation
