"""Trains neural network to assess the quality of photos."""
import datetime
import os
import numpy
import tensorflow as tf

from flickr_dataset_loader import FlickrDatasetLoader
from neural_network_model import NeuralNetworkModel

# TODO save the model after training and write a script to process a given photo

# logging configuration
LOG_DEVICE_PLACEMENT = False

# training parameters
TRAINING_BATCH_SIZE = 10
TRAINING_ITERATIONS_LIMIT = 1000000000
NUMBER_OF_EPOCHS = 10

# evaluation parameters
TEST_BATCH_SIZE = 10
VALIDATION_BATCH_SIZE = 10
VALIDATION_SET_SIZE_LIMIT = 1000000
TEST_SET_SIZE_LIMIT = 10000000

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
  training_set_predictions = neural_network_model.generate_oxford_network_model(training_photo_batch)
  training_label_batch = tf.Print(training_label_batch, [training_label_batch], message="Labels: ", summarize=10)
  # training_error = tf.nn.sigmoid_cross_entropy_with_logits(training_set_predictions, training_label_batch)
  # training_error = tf.reduce_mean(cross_entropy)
  training_error = tf.nn.l2_loss(tf.sub(training_set_predictions, training_label_batch))
  training_error = tf.Print(training_error, [training_error], message="Error on the training batch: ", summarize=10)
  train_step = tf.train.AdamOptimizer().minimize(training_error)

  # evaluating on training set
  training_correct_prediction = tf.equal(tf.round(training_set_predictions), training_label_batch)
  training_correct_prediction = tf.Print(training_correct_prediction, [training_correct_prediction],
                                         message="Correct predictions: ", summarize=10)
  training_accuracy = tf.reduce_mean(tf.cast(training_correct_prediction, tf.float32))

  # evaluating on validation set
  validation_set_predictions = neural_network_model.generate_oxford_network_model(validation_photo_batch, True)
  validation_label_batch = tf.Print(validation_label_batch, [validation_label_batch], message="Labels: ", summarize=10)
  validation_correct_prediction = tf.equal(tf.round(validation_set_predictions), validation_label_batch)
  validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_prediction, tf.float32))
  # validation_error = tf.reduce_mean(
  #   tf.nn.sigmoid_cross_entropy_with_logits(validation_set_predictions, validation_label_batch))
  validation_error = tf.nn.l2_loss(tf.sub(validation_set_predictions, validation_label_batch))

  # evaluating on test set
  test_set_predictions = neural_network_model.generate_oxford_network_model(test_photo_batch, True)
  test_correct_prediction = tf.equal(tf.round(test_set_predictions), test_label_batch)
  test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

  # initialize session
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

  # initialize TensorBoard
  with tf.name_scope('input'):
    input_images_summary = tf.summary.image('images', training_photo_batch, 20)
  with tf.name_scope('evaluation'):
    training_accuracy_summary = tf.summary.scalar('Accuracy on the training set', training_accuracy)
    training_error_summary = tf.summary.scalar('Error on the training set', training_error)
  training_writer = tf.summary.FileWriter(TRAINING_SUMMARY_DIR, sess.graph)
  validation_writer = tf.summary.FileWriter(VALIDATION_SUMMARY_DIR, sess.graph)
  test_writer = tf.summary.FileWriter(TEST_SUMMARY_DIR, sess.graph)

  # do the training
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  number_of_batches = flickr_dataset_loader.get_training_set_size() // TRAINING_BATCH_SIZE
  number_of_iterations = min(number_of_batches * NUMBER_OF_EPOCHS, TRAINING_ITERATIONS_LIMIT)
  for i in range(number_of_iterations):
    print('Training with batch {}/{} (next {} photos)...'.format(i + 1, number_of_iterations, TRAINING_BATCH_SIZE))
    _, cross_entropy_result, training_accuracy_result = sess.run(
      [train_step, training_error_summary, training_accuracy_summary])
    training_writer.add_summary(training_accuracy_result, i)
    training_writer.add_summary(cross_entropy_result, i)
    # training_writer.add_summary(image_result, i)

    # evaluate the model on validation set every 50th iteration
    if i % 50 == 0:
      print('Evaluating the model on the cross-validation set...')
      accuracies = []
      validation_error_values = []
      for _ in range(
              min(flickr_dataset_loader.get_validation_set_size(), VALIDATION_SET_SIZE_LIMIT) // VALIDATION_BATCH_SIZE):
        validation_accuracy_result, validation_error_result = sess.run(
          [validation_accuracy, validation_error])
        accuracies.append(validation_accuracy_result)
        validation_error_values.append(validation_error_result)
      average_validation_accuracy = numpy.asscalar(numpy.mean(accuracies))
      average_validation_error = numpy.asscalar(numpy.mean(validation_error_values))
      with tf.name_scope('evaluation'):
        validation_accuracy_result = tf.Summary()
        validation_accuracy_result.value.add(tag="Accuracy on the cross-validation set",
                                             simple_value=average_validation_accuracy)
        validation_cross_entropy_result = tf.Summary()
        validation_cross_entropy_result.value.add(tag="Average error on the cross-validation set",
                                                  simple_value=average_validation_error)
      validation_writer.add_summary(validation_accuracy_result, i)
      validation_writer.add_summary(validation_cross_entropy_result, i)

  # compute accuracy of the trained model on test set
  print('Done! The whole training took {} seconds.'.format((datetime.datetime.now() - start_time).seconds))
  test_accuracies = []
  for _ in range(min(flickr_dataset_loader.get_test_set_size(), TEST_SET_SIZE_LIMIT) // TEST_BATCH_SIZE):
    test_accuracies.append(sess.run(test_accuracy))
  average_test_accuracy = numpy.asscalar(numpy.mean(test_accuracies))
  with tf.name_scope('evaluation'):
    test_accuracy_result = tf.Summary()
    test_accuracy_result.value.add(tag="Accuracy on the test set", simple_value=average_test_accuracy)
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
