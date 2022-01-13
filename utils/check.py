import numpy as np
import sys
import pathlib
sys.path.insert(1, "..")
# from test import test

import inspect, os
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir)) + '\data'
# print(currentdir, parentdir)

# data_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
# print(data_path)

# def createNPArray():
#     return np.array([1, 2, 3, 4, 5])

from glob import glob
def globbing():
    print(glob('./trained_models/*.pth'))

def paths():
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    data_path = os.path.dirname(parent_path) + '/data'
    print(parent_path)
    print(data_path)

paths()