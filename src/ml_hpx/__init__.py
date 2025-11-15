import atexit

from ._ml_hpx_impl import (
    SGD,
    SVC,
    KMeansClustering,
    Layer,
    NeuralNetwork,
    Optimizer,
    Perceptron,
    finalize,
    initialize,
)

initialize()

atexit.register(finalize)
