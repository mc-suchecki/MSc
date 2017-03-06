import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib

# settings
TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X_fc6_cut.npy'
LABELS_FILE_NAME = 'y_fc6_cut.npy'
TUNED_PARAMETERS = [
  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
  {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
]
SCORES = ['accuracy']

if __name__ == '__main__':
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

  # grid search for optimal hyper-parameters
  for score in SCORES:
    print('Tuning hyper-parameters for {}'.format(score))

    # training (increasing cache size is recommended with large RAM)
    classifier = GridSearchCV(SVC(cache_size=512), TUNED_PARAMETERS, scoring=score, n_jobs=4, verbose=3)
    classifier.fit(X_train, y_train)
    print()

    # results
    print('Best parameters found on the training set:')
    print(classifier.best_params_)
    print()
    print('Grid scores on the training set:')
    print()
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
      print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()
    predictions = classifier.predict(X_test)
    print('Classification report on the test set:')
    print(classification_report(y_test, predictions))
    print()
    print('Confusion matrix for the test set:')
    print(confusion_matrix(y_test, predictions))
    print()

    # saving the model
    print('Saving the model...')
    model_file_name = './results/svm_fc6_{}_model.pkl'.format(len(y_train))
    joblib.dump(classifier, model_file_name)
    print('Model saved to {}.'.format(model_file_name))
