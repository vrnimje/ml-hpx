from ._ml_hpx_impl import LinearRegression, LogisticRegression, KNearestNeighbours, KMeansClustering
from ._ml_hpx_impl import finalize, initialize
import atexit

initialize()

atexit.register(finalize)
