import numpy
from sklearn import svm

# settings
TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X.npy'
LABELS_FILE_NAME = 'y.npy'

# training
print("Loading training data...")
X = numpy.load(TRAIN_DATA_LOCATION + EXAMPLES_FILE_NAME)
y = numpy.load(TRAIN_DATA_LOCATION + LABELS_FILE_NAME)
print("X shape is {}, y shape is {}".format(X.shape, y.shape))
model = svm.SVC()
print("Training the SVM classifier...")
model.fit(X, y)

# testing
print("Loading test data...")
X = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print("X shape is {}, y shape is {}".format(X.shape, y.shape))
print("Testing the SVM classifier...")
score = model.score(X, y)
print("Mean accuracy on the test set is {}.".format(score))
