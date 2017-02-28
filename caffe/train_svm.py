import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

# settings
TRAIN_DATA_LOCATION = '../data/train/'
TEST_DATA_LOCATION = '../data/test/'
EXAMPLES_FILE_NAME = 'X_cut.npy'
LABELS_FILE_NAME = 'y_cut.npy'
TUNED_PARAMETERS = {'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}
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

    # increasing cache size is recommended with large RAM
    classifier = GridSearchCV(SVC(cache_size=512), TUNED_PARAMETERS, scoring=score, n_jobs=-1, verbose=3)
    classifier.fit(X_train, y_train)
    print()

    print('Best parameters found on the training set:')
    print(classifier.best_params_)
    print()
    print('Grid scores on the training set:')
    print()
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
      print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print('Classification report on the test set:')
    print(classification_report(y_test, classifier.predict(X_test)))
    print()

    # saving the model
    print('Saving the model...')
    joblib.dump(classifier, './svm_rbf_grid_search_model_best_{}.pkl'.format(score))
