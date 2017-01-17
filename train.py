"""Trains neural network to assess the quality of photos."""
import datetime
import os
import numpy
import tensorflow as tf

from flickr_dataset_loader import FlickrDatasetLoader
from neural_network_model import NeuralNetworkModel

# TODO save the model after training and write a script to process a given photo

# TensorFlow configuration
LOG_DEVICE_PLACEMENT = True

# training parameters
TRAINING_BATCH_SIZE = 10
TRAINING_ITERATIONS_LIMIT = 100000

# evaluation parameters
TEST_BATCH_SIZE = 10
VALIDATION_BATCH_SIZE = 10
VALIDATION_SET_SIZE = 1500  # TODO this should not be here
TEST_SET_SIZE = 1500  # TODO this should not be here

# TensorBoard configuration
TENSOR_BOARD_DIR = os.path.join('.', 'tmp')
TRAINING_SUMMARY_DIR = './tmp/train'
VALIDATION_SUMMARY_DIR = './tmp/validation'
TEST_SUMMARY_DIR = './tmp/test'


def main(_):
  # clean TensorBoard data directory
  if tf.gfile.Exists(TENSOR_BOARD_DIR):
    tf.gfile.DeleteRecursively(TENSOR_BOARD_DIR)
  tf.gfile.MakeDirs(TENSOR_BOARD_DIR)

  # start measuring time
  start_time = datetime.datetime.now()

  # load the datasets
  flickr_dataset_loader = FlickrDatasetLoader()
  training_photo_batch, training_label_batch = flickr_dataset_loader.create_training_batch(TRAINING_BATCH_SIZE)
  validation_photo_batch, validation_label_batch = flickr_dataset_loader.create_validation_batch(VALIDATION_BATCH_SIZE)
  test_photo_batch, test_label_batch = flickr_dataset_loader.create_test_batch(TEST_BATCH_SIZE)

  # training
  neural_network_model = NeuralNetworkModel()
  training_set_predictions = neural_network_model.generate_network_model(training_photo_batch)
  cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(training_set_predictions, tf.cast(training_label_batch, tf.float32)))
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

  # evaluating on training set
  training_correct_prediction = tf.equal(tf.cast(tf.round(training_set_predictions), tf.bool), training_label_batch)
  training_accuracy = tf.reduce_mean(tf.cast(training_correct_prediction, tf.float32))

  # evaluating on validation set
  validation_set_predictions = neural_network_model.generate_network_model(validation_photo_batch, True)
  validation_correct_prediction = tf.equal(tf.cast(tf.round(validation_set_predictions), tf.bool),
                                           validation_label_batch)
  validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_prediction, tf.float32))

  # evaluating on test set
  test_set_predictions = neural_network_model.generate_network_model(test_photo_batch, True)
  test_correct_prediction = tf.equal(tf.cast(tf.round(test_set_predictions), tf.bool), test_label_batch)
  test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

  # initialize session
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

  # initialize TensorBoard
  # with tf.name_scope('input'):
  #   input_images_summary = tf.summary.image('images', training_photo_batch, 20)
  with tf.name_scope('Model evaluation'):
    training_accuracy_summary = tf.summary.scalar('Accuracy on the training set', training_accuracy)
  training_writer = tf.summary.FileWriter(TRAINING_SUMMARY_DIR, sess.graph)
  validation_writer = tf.summary.FileWriter(VALIDATION_SUMMARY_DIR, sess.graph)
  test_writer = tf.summary.FileWriter(TEST_SUMMARY_DIR, sess.graph)

  # do the training
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  number_of_batches = flickr_dataset_loader.get_training_set_size() // TRAINING_BATCH_SIZE
  for i in range(min(number_of_batches, TRAINING_ITERATIONS_LIMIT)):
    print('Training with batch {}/{} (containing {} photos)...'.format(i + 1, number_of_batches, TRAINING_BATCH_SIZE))
    training_accuracy_result, _ = sess.run([training_accuracy_summary, train_step])
    training_writer.add_summary(training_accuracy_result, i)
    # training_writer.add_summary(input_images_summary, i)

    # evaluate the model on validation set every 20th iteration
    if i % 20 == 0:
      print('Evaluating the model on the cross-validation set...')
      accuracies = []
      for _ in range(VALIDATION_SET_SIZE // VALIDATION_BATCH_SIZE):
        accuracies.append(sess.run(validation_accuracy))
      average_validation_accuracy = numpy.asscalar(numpy.mean(accuracies))
      with tf.name_scope('Model evaluation'):
        validation_accuracy_result = tf.Summary()
        validation_accuracy_result.value.add(name="Accuracy on the cross-validation set",
                                             simple_value=average_validation_accuracy)
      validation_writer.add_summary(validation_accuracy_result, i)

  # compute accuracy of the trained model on test set
  print('Done! The whole training took {} seconds.'.format((datetime.datetime.now() - start_time).seconds))
  test_accuracies = []
  for _ in range(VALIDATION_SET_SIZE // VALIDATION_BATCH_SIZE):
    test_accuracies.append(sess.run(test_accuracy))
  average_test_accuracy = numpy.asscalar(numpy.mean(test_accuracies))
  with tf.name_scope('evaluation'):
    test_accuracy_result = tf.Summary()
    test_accuracy_result.value.add(name="Accuracy on the test set", simple_value=average_test_accuracy)
  test_writer.add_summary(test_accuracy_result)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  training_writer.close()
  validation_writer.close()
  test_writer.close()
  sess.close()


if __name__ == '__main__':
  tf.app.run()
