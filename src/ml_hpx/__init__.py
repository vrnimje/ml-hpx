from ._ml_hpx_impl import (
    LinearRegression,
    LogisticRegression,
    KNearestNeighbours,
    KMeansClustering,
    Perceptron,
    SVC,
)
from ._ml_hpx_impl import finalize, initialize
from ._ml_hpx_impl import NeuralNetwork, Layer, Optimizer, SGD

import atexit


def safe_initialize():
    try:
        initialize()
    except OSError as e:
        if "libnvidia" in str(e):
            # Ignore missing GPU libraries on CPU-only systems
            pass
        else:
            raise


safe_initialize()

initialize()

atexit.register(finalize)
