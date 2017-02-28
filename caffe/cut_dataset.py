import numpy

TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X.npy'
LABELS_FILE_NAME = 'y.npy'
CUT_EXAMPLES_FILE_NAME = 'X_cut.npy'
CUT_LABELS_FILE_NAME = 'y_cut.npy'
TRAINING_SET_SIZE = 2**17
TEST_SET_SIZE = 2**15

print('Loading training data...')
X_train = numpy.load(TRAIN_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_train = numpy.load(TRAIN_DATA_LOCATION + LABELS_FILE_NAME)
print('Training data: X shape is {}, y shape is {}'.format(X_train.shape, y_train.shape))

print('Cutting training data...')
X_train = X_train[:TRAINING_SET_SIZE:1, ::1]
y_train = y_train[:TRAINING_SET_SIZE:1]
print('Training data: X shape is {}, y shape is {}'.format(X_train.shape, y_train.shape))

print('Saving training data...')
numpy.save(TRAIN_DATA_LOCATION + CUT_EXAMPLES_FILE_NAME, X_train)
numpy.save(TRAIN_DATA_LOCATION + CUT_LABELS_FILE_NAME, y_train)

print('Loading test data...')
X_test = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_test = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print('Test data: X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))

print('Cutting test data...')
X_test = X_test[:TEST_SET_SIZE:1, ::1]
y_test = y_test[:TEST_SET_SIZE:1]
print('Test data: X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))

print('Saving test data...')
numpy.save(TEST_DATA_LOCATION + CUT_EXAMPLES_FILE_NAME, X_test)
numpy.save(TEST_DATA_LOCATION + CUT_LABELS_FILE_NAME, y_test)

