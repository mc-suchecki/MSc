import numpy
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# settings
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X_fc6_no_relu_full.npy'
LABELS_FILE_NAME = 'y_fc6_no_relu_full.npy'

# loading
print('Loading the classifier...')
classifier = joblib.load('./results/svm_fc6_65536_model.pkl')

# testing
print('Loading test data...')
X_test = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_test = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print('X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))
print('Testing the classifier...')
predictions = classifier.predict(X_test)
print('Saving the predictions...')
numpy.save('./results/svm_predictions.npy', predictions)
print('Making sure that both lists are of the same type...')
predictions = [int(label) for label in predictions]
y_test = [int(label) for label in y_test]
print('Classification report on the test set:')
print(classification_report(y_test, predictions))
print()
print('Confusion matrix for the test set:')
print(confusion_matrix(y_test, predictions))
print()
