import numpy
import tensorflow as tf
from tensorflow.python.framework import ops


class FlickrDatasetLoader(object):
  """ A class loading the photos from Flickr along with the associated labels. """

  # photos metadata
  IMAGE_WIDTH = 240
  """ Width in pixels of the photos to import. Must be the same across all photos. """
  IMAGE_HEIGHT = 159
  """ Height in pixels of the photos to import. Must be the same across all photos. Currently set to 180 pixels, as
      horizontal images with 4:3 aspect ratio are the most common ones. """
  NUMBER_OF_CHANNELS = 3
  """ Number of channels in loaded photos. All images are RGB images, thus there are 3 channels. """
  TRAINING_EXAMPLE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_CHANNELS
  """ Size of the training example. Equals to width * height * number of channels. """

  # information about the files location
  SOURCE_DIR = '/media/p307k07/ssd/opt/msc/data/'
  """ Location of the directory with the Flickr dataset. """
  TRAINING_IMAGES_DIR = SOURCE_DIR + 'train/'
  """ Relative location of the directory with the training set. """
  VALIDATION_IMAGES_DIR = SOURCE_DIR + 'validation/'
  """ Relative location of the directory with the cross validation set. """
  TEST_IMAGES_DIR = SOURCE_DIR + 'test/'
  """ Relative location of the directory with the test set. """
  LIST_FILE_NAME = 'list.txt'
  """ Name of the file which contains a list of photos filenames with their labels for each dataset (train/cv/test). """
  MEAN_IMAGE_FILE_NAME = 'mean.npy'
  """ Name of the file which contains a NumPy array with mean values for each pixel for every photo in the appropriate
  dataset. Used for normalization purposes. """

  def __init__(self):
    self.training_set_size = 0
    self.validation_set_size = 0
    self.test_set_size = 0

  def create_training_batch(self, batch_size: int = None):
    """ Creates batch of photos and corresponding labels from the training set. """
    training_batch, training_set_size = self._create_photo_and_label_batches(self.TRAINING_IMAGES_DIR, batch_size)
    # save the size of the loaded dataset
    self.training_set_size = training_set_size
    return training_batch

  def create_validation_batch(self, batch_size: int = None):
    """ Creates batch of photos and corresponding labels from the cross validation set. """
    validation_batch, validation_set_size = self._create_photo_and_label_batches(self.VALIDATION_IMAGES_DIR, batch_size)
    # save the size of the loaded dataset
    self.validation_set_size = validation_set_size
    return validation_batch

  def create_test_batch(self, batch_size: int = None):
    """ Creates batch of photos and corresponding labels from the training set. """
    test_batch, test_set_size = self._create_photo_and_label_batches(self.TEST_IMAGES_DIR, batch_size)
    # save the size of the loaded dataset
    self.test_set_size = test_set_size
    return test_batch

  def _create_photo_and_label_batches(self, source_directory: str, batch_size: int):
    # TODO split this function into smaller functions
    # read the list of photo IDs and labels
    photos_list = open(source_directory + self.LIST_FILE_NAME)
    filenames_list = []
    labels_list = []
    # get lists of photo file names and labels
    for line in photos_list:
      filenames_list.append(source_directory + line.split(',')[0] + '.jpg')
      labels_list.append([bool(int(line.split(',')[1]))])
    # set batch size to the whole dataset if batch size is not provided
    if batch_size is None:
      batch_size = len(filenames_list)
    # convert the lists to tensors
    filenames = tf.convert_to_tensor(filenames_list, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_list, dtype=tf.float32)
    with ops.name_scope(None, 'load_input_data', [filenames, labels]):
      # create a constant holding the mean values for every pixel
      mean = tf.constant(numpy.load(source_directory + self.MEAN_IMAGE_FILE_NAME), dtype=tf.float32)
      # create queue with filenames and labels
      file_name, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
      # file_name = tf.Print(file_name, [file_name], message="Opening photo: ")
      # convert filenames of photos to input vectors
      photo = tf.read_file(file_name)
      uint8_color_values = tf.image.decode_jpeg(photo, channels=self.NUMBER_OF_CHANNELS)
      uint8_color_values.set_shape([self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUMBER_OF_CHANNELS])
      float32_color_values = tf.to_float(uint8_color_values)
      # subtract the mean (dataset normalization)
      float32_color_values = float32_color_values - mean
      # slice the data into mini batches
      return tf.train.batch([float32_color_values, label], batch_size=batch_size), len(filenames_list)

  def get_training_set_size(self):
    if self.training_set_size == 0:
      raise AssertionError('Training set was not loaded yet!')
    else:
      return self.training_set_size

  def get_validation_set_size(self):
    if self.validation_set_size == 0:
      raise AssertionError('Cross validation set was not loaded yet!')
    else:
      return self.validation_set_size

  def get_test_set_size(self):
    if self.test_set_size == 0:
      raise AssertionError('Test set was not loaded yet!')
    else:
      return self.test_set_size
