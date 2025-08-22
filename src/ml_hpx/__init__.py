from ._ml_hpx_impl import LinearRegression, LogisticRegression, KNearestNeighbours, KMeansClustering, Perceptron, SVC
from ._ml_hpx_impl import finalize, initialize
from ._ml_hpx_impl import NeuralNetwork, Layer, Optimizer, SGD

import atexit

initialize()

atexit.register(finalize)
