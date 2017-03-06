import numpy
from sklearn.externals import joblib

# settings
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X.npy'
LABELS_FILE_NAME = 'y.npy'

# loading
print("Loading the model...")
model = joblib.load('./random_forests_model.pkl')

# testing
print("Loading test data...")
X = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print("X shape is {}, y shape is {}".format(X.shape, y.shape))
print("Testing the SVM classifier...")
score = model.score(X, y)
print("Mean accuracy on the test set is {}.".format(score))
