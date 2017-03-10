import numpy

FILES_EXTENSION = '.npy'
TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X_fc6_no_relu_first_65536'
LABELS_FILE_NAME = 'y_fc6_no_relu_first_65536'
TRAINING_SET_SIZE = 2 ** 14
TEST_SET_SIZE = 2 ** 12
TRAINING_SET_SUFFIX = '_first_' + str(TRAINING_SET_SIZE) + FILES_EXTENSION
TEST_SET_SUFFIX = '_first_' + str(TEST_SET_SIZE) + FILES_EXTENSION

print('Loading training data...')
X_train = numpy.load(TRAIN_DATA_LOCATION + EXAMPLES_FILE_NAME + FILES_EXTENSION)
y_train = numpy.load(TRAIN_DATA_LOCATION + LABELS_FILE_NAME + FILES_EXTENSION)
print('Training data: X shape is {}, y shape is {}'.format(X_train.shape, y_train.shape))

print('Cutting training data...')
X_train = X_train[:TRAINING_SET_SIZE:1, ::1]
y_train = y_train[:TRAINING_SET_SIZE:1]
print('Training data: X shape is {}, y shape is {}'.format(X_train.shape, y_train.shape))

print('Saving training data...')
numpy.save(TRAIN_DATA_LOCATION + 'X_fc6_no_relu' + TRAINING_SET_SUFFIX, X_train)
print('Training examples saved to:' + TRAIN_DATA_LOCATION + EXAMPLES_FILE_NAME + TRAINING_SET_SUFFIX)
numpy.save(TRAIN_DATA_LOCATION + 'y_fc6_no_relu' + TRAINING_SET_SUFFIX, y_train)
print('Training labels saved to:' + TRAIN_DATA_LOCATION + LABELS_FILE_NAME + TRAINING_SET_SUFFIX)

print('Loading test data...')
X_test = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_test = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print('Test data: X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))

print('Cutting test data...')
X_test = X_test[:TEST_SET_SIZE:1, ::1]
y_test = y_test[:TEST_SET_SIZE:1]
print('Test data: X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))

print('Saving test data...')
numpy.save(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME + TEST_SET_SUFFIX, X_test)
print('Test examples saved to:' + TEST_DATA_LOCATION + EXAMPLES_FILE_NAME + TEST_SET_SUFFIX)
numpy.save(TEST_DATA_LOCATION + LABELS_FILE_NAME + TEST_SET_SUFFIX, y_test)
print('Test labels saved to:' + TEST_DATA_LOCATION + LABELS_FILE_NAME + TEST_SET_SUFFIX)
