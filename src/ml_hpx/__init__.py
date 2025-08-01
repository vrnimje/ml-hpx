from ._ml_hpx_impl import LinearRegression, LogisticRegression, KNearestNeighbours, finalize, initialize
import atexit

initialize()

atexit.register(finalize)
