import numpy
from pandas import DataFrame

X = numpy.load('../data/train/X_fc6_cut.npy')
data = DataFrame(data=X)
print(data.describe())

