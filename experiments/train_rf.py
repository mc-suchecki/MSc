import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# settings
TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X_fc6_cut.npy'
LABELS_FILE_NAME = 'y_fc6_cut.npy'

# loading the dataset
print('Loading training data...')
X_train = numpy.load(TRAIN_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_train = numpy.load(TRAIN_DATA_LOCATION + LABELS_FILE_NAME)
print('Training data: X shape is {}, y shape is {}'.format(X_train.shape, y_train.shape))
print('Loading test data...')
X_test = numpy.load(TEST_DATA_LOCATION + EXAMPLES_FILE_NAME)
y_test = numpy.load(TEST_DATA_LOCATION + LABELS_FILE_NAME)
print('Test data: X shape is {}, y shape is {}'.format(X_test.shape, y_test.shape))
print()

# training
print('Training random forests classifier...')
classifier = RandomForestClassifier(n_estimators=10)
classifier = classifier.fit(X_train, y_train)

# results
predictions = classifier.predict(X_test)
print('Classification report on the test set:')
print(classification_report(y_test, predictions))
print()
print('Confusion matrix for the test set:')
print(confusion_matrix(y_test, predictions))
print()

# saving the model
print('Saving the model...')
model_file_name = './results/rf_fc6_{}_model.pkl'.format(len(y_train))
joblib.dump(classifier, model_file_name)
print('Model saved to {}.'.format(model_file_name))

