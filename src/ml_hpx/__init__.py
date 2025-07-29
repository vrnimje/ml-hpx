from ._ml_hpx_impl import LinearRegression, LogisticRegression, finalize, initialize
import atexit

initialize()

atexit.register(finalize)
