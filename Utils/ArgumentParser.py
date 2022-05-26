import os
from typing import Union


class ArgumentParser:

    # Static variable of the ArgumentParser class that lists the supported argument types
    __acceptedTypes = ["str", "int", "float", "bool", "path"]


    # Utility function to determine whether a String is an Integer
    @staticmethod
    def __isInt(argument: str):
        try:
            int(argument)
            return True
        except:
            return False

    # Utility function to determine whether a String is a Float
    @staticmethod
    def __isFloat(argument: str):
        try:
            float(argument)
            return True
        except:
            return False

    # Utility function to determine whether a String is a Boolean
    @staticmethod
    def __isBool(argument: str):
        return argument in ["0", "1", "True", "true", "False", "false"]

    # Utility function to determine whether a File exists
    @staticmethod
    def __FileExists(argument: str):
        return os.path.isfile(argument)

    def __init__(self):
        self.__arguments = {}
        self.__argumentsParsed = False
        self.__mandatoryArguments = 0
        self.__optionalArguments = 0

    # Utility function to add an argument of any form to the corresponding ArgumentParser object
    def addArgument(self, argument: str, type: str, mandatory: bool = True):
        if argument not in self.__arguments and type in ArgumentParser.__acceptedTypes and not self.__argumentsParsed:
            if type in ["float", "int"]:
                self.__arguments[argument] = [type, mandatory, None, None]
            else:
                self.__arguments[argument] = [type, mandatory]
            if mandatory:
                self.__mandatoryArguments += 1
            else:
                self.__optionalArguments += 1

    # Utility function to add a numeric argument to the corresponding ArgumentParser object
    def addNumericArgument(self, argument: str, type: str, floor: Union[float, int] = None, ceiling: Union[float, int] = None, mandatory: bool = True):
        if floor is not None and ceiling is not None and floor > ceiling:
            pass
        elif argument not in self.__arguments and type in ["float", "int"] and not self.__argumentsParsed:
            self.__arguments[argument] = [type, mandatory, floor, ceiling]

    # Utility function to parse the provided arguments
    def parse(self, args: list):
        self.__argumentsParsed = True

        # Say X is |args| - 1 then X should be divisible by 2 and greater than or equal to (2 * mandatory arguments)
        if (len(args) - 1) % 2 != 0 or ((self.__mandatoryArguments * 2) > (len(args) - 1)):
            print("Error : Invalid number of arguments were provided")
            return False

        for argument in self.__arguments:

            # In case a mandatory argument was not provided
            if argument not in args and self.__arguments[argument][1]:
                print("Error : Argument {} was not provided".format(argument))
                return False

            # In case an optional argument was not provided
            if argument not in args and not self.__arguments[argument][1]:
                continue

            argumentType = self.__arguments[argument][0]
            argumentValue = args[args.index(argument) + 1]

            # Trivial checks to decide whether the corresponding "int" argument can be parsed
            if argumentType == "int":
                floor = self.__arguments[argument][2]
                ceiling = self.__arguments[argument][3]

                if not ArgumentParser.__isInt(argumentValue):
                    print("Error : Argument {} should be an integer value".format(argument))
                    return False

                elif floor is not None and ceiling is None and int(argumentValue) < floor:
                    print("Error : Argument {} should be an integer value greater than or equal to {}".format(argument, floor))
                    return False

                elif floor is None and ceiling is not None and int(argumentValue) > ceiling:
                    print("Error : Argument {} should be an integer value less than or equal to {}".format(argument, ceiling))
                    return False

                elif floor is not None and ceiling is not None and (int(argumentValue) > ceiling or int(argumentValue) < floor):
                    print("Error : Argument {} should be an integer value in the following range [{},{}]".format(argument, floor, ceiling))
                    return False

                else:
                    self.__arguments[argument] = int(argumentValue)

            # Trivial checks to decide whether the corresponding "float" argument can be parsed
            elif argumentType == "float":
                floor = self.__arguments[argument][2]
                ceiling = self.__arguments[argument][3]

                if not ArgumentParser.__isFloat(argumentValue):
                    print("Error : Argument {} should be a real number".format(argument))
                    return False

                elif floor is not None and ceiling is None and float(argumentValue) < floor:
                    print("Error : Argument {} should be a real number greater than or equal to {}".format(argument, floor))
                    return False

                elif floor is None and ceiling is not None and float(argumentValue) > ceiling:
                    print("Error : Argument {} should be a real number less than or equal to {}".format(argument, ceiling))
                    return False

                elif floor is not None and ceiling is not None and (float(argumentValue) > ceiling or float(argumentValue) < floor):
                    print("Error : Argument {} should be a real number in the following range [{},{}]".format(argument, floor, ceiling))
                    return False

                else:
                    self.__arguments[argument] = float(argumentValue)

            # Trivial checks to decide whether the corresponding "bool" argument can be parsed
            elif argumentType == "bool":

                if ArgumentParser.__isBool(argumentValue):
                    if argumentValue in ["1", "true", "True"]:
                        self.__arguments[argument] = True
                    else:
                        self.__arguments[argument] = False

                else:
                    print("Error : Argument {} should be a boolean value".format(argument))
                    return False

            # Trivial checks to decide whether the corresponding "path" argument can be parsed
            elif argumentType == "path":

                if ArgumentParser.__FileExists(argumentValue):
                    self.__arguments[argument] = argumentValue

                else:
                    print("Error : Argument {} should the path of an existing file".format(argument))
                    return False

            else:
                self.__arguments[argument] = argumentValue

        return True

    # Utility function to get an argument under the constraint that the .parse() function has been called in the past
    def getArgument(self, argument: str):
        if self.__argumentsParsed and argument in self.__arguments:
            if type(self.__arguments[argument]) == list:
                return None
            else:
                return self.__arguments[argument]
        else:
            return None

    def clear(self):
        if self.__argumentsParsed:
            self.__argumentsParsed = False
            self.__mandatoryArguments = 0
            self.__optionalArguments = 0
            self.__arguments.clear()
