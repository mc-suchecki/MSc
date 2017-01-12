"""Trains neural network to assess the quality of photos."""
import datetime
import os
import tensorflow as tf

from flickr_dataset_loader import FlickrDatasetLoader

# TODO use placeholders for list of filenames in order to not duplicate the model code 3 times
# TODO re-think the computation graph and add namespaces
# TODO save the model after training and write a script to process a given photo
# TODO add convolution layers and test bigger models

# training parameters
TRAINING_BATCH_SIZE = 100
TRAINING_ITERATIONS_LIMIT = 10

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
  validation_photo_batch, validation_label_batch = flickr_dataset_loader.create_validation_batch()
  test_photo_batch, test_label_batch = flickr_dataset_loader.create_test_batch()

  # create the model
  x = training_photo_batch  # input tensor (so far flattened)
  w = tf.Variable(tf.zeros([flickr_dataset_loader.TRAINING_EXAMPLE_SIZE, 1], dtype=tf.float32))  # weights tensor
  b = tf.Variable(tf.zeros([1], dtype=tf.float32))  # bias tensor (so far only 1 neuron)
  y_ = training_label_batch  # labels tensor
  y = tf.sigmoid(tf.matmul(x, w) + b)  # predictions

  # training
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

  # evaluating on training set
  training_correct_prediction = tf.equal(tf.cast(tf.round(y), tf.bool), training_label_batch)
  training_accuracy = tf.reduce_mean(tf.cast(training_correct_prediction, tf.float32))

  # evaluating on validation set
  y_validation = tf.sigmoid(tf.matmul(validation_photo_batch, w) + b)
  validation_correct_prediction = tf.equal(tf.cast(tf.round(y_validation), tf.bool), validation_label_batch)
  validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_prediction, tf.float32))

  # evaluating on test set
  y_test = tf.sigmoid(tf.matmul(test_photo_batch, w) + b)
  test_correct_prediction = tf.equal(tf.cast(tf.round(y_test), tf.bool), test_label_batch)
  test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

  # initialize session
  sess = tf.InteractiveSession()

  # initialize TensorBoard
  # with tf.name_scope('input'):
  #   tf.summary.image('images', x, 20)
  with tf.name_scope('evaluation'):
    training_accuracy_summary = tf.summary.scalar('Accuracy on the training set', training_accuracy)
    validation_accuracy_summary = tf.summary.scalar('Accuracy on the cross validation set', validation_accuracy)
    test_accuracy_summary = tf.summary.scalar('Accuracy on the test set', test_accuracy)
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
    print('Evaluating the model on validation set...')
    validation_accuracy_result = sess.run(validation_accuracy_summary)
    validation_writer.add_summary(validation_accuracy_result, i)

  # compute accuracy of the trained model on test set
  print('Done! The whole training took {} seconds.'.format((datetime.datetime.now() - start_time).seconds))
  test_accuracy_result = sess.run(test_accuracy_summary)
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
