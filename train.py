"""Trains neural network to assess the quality of photos."""
import datetime
import os
import tensorflow as tf

from flickr_dataset_loader import FlickrDatasetLoader

# TODO re-think the computation graph and add namespaces
# TODO improve evaluation on the validation set - so far we only evaluate on 100 images
# TODO evaluate the model on the test set at the end
# TODO find a way to plot accuracy on the training set also?
# TODO save the model after training and write a script to process a given photo

# batch sizes
TRAINING_BATCH_SIZE = 100
VALIDATION_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

# TensorBoard configuration
TENSOR_BOARD_DIR = os.path.join('.', 'tmp')


def main(_):
  # clean TENSOR_BOARD_DIR
  if tf.gfile.Exists(TENSOR_BOARD_DIR):
    tf.gfile.DeleteRecursively(TENSOR_BOARD_DIR)
  tf.gfile.MakeDirs(TENSOR_BOARD_DIR)

  # start measuring time
  start_time = datetime.datetime.now()

  # load the datasets
  flickr_dataset_loader = FlickrDatasetLoader()
  training_photo_batch, training_label_batch = flickr_dataset_loader.create_training_batch(TRAINING_BATCH_SIZE)
  validation_photo_batch, validation_label_batch = flickr_dataset_loader.create_validation_batch(VALIDATION_BATCH_SIZE)
  # test_photo_batch, test_label_batch = flickr_dataset_loader.create_test_batch(TEST_BATCH_SIZE)

  # create the model
  x = training_photo_batch  # input tensor (so far flattened)
  w = tf.Variable(tf.zeros([flickr_dataset_loader.TRAINING_EXAMPLE_SIZE, 1], dtype=tf.float32))  # weights tensor
  b = tf.Variable(tf.zeros([1], dtype=tf.float32))  # bias tensor (so far only 1 neuron)
  y_ = training_label_batch  # labels tensor
  y = tf.sigmoid(tf.matmul(x, w) + b)  # predictions

  # training
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # evaluating
  y_validation = tf.sigmoid(tf.matmul(validation_photo_batch, w) + b)
  correct_prediction = tf.equal(tf.cast(tf.round(y_validation), tf.bool), validation_label_batch)
  validation_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # initialize session
  sess = tf.InteractiveSession()

  # initialize TensorBoard
  # with tf.name_scope('input'):
  #   tf.summary.image('images', x, 20)
  with tf.name_scope('evaluation'):
    tf.summary.scalar('accuracy', validation_accuracy)
  summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./tmp/train', sess.graph)
  validation_writer = tf.summary.FileWriter('./tmp/validation', sess.graph)

  # do the training
  tf.initialize_all_variables().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  number_of_batches = flickr_dataset_loader.get_training_set_size() // TRAINING_BATCH_SIZE
  for i in range(number_of_batches):
    print('Training with batch {}/{} (containing {} photos)...'.format(i, number_of_batches, TRAINING_BATCH_SIZE))
    # summary, _ = sess.run([summaries, train_step])
    sess.run(train_step)
    # train_writer.add_summary(summary, i)
    print('Evaluating the model on validation set...')
    # TODO this is evaluating only 100 images from the validation set!
    validation_accuracy_summary = sess.run(summaries)
    validation_writer.add_summary(validation_accuracy_summary, i)

  # test trained model
  print('Done! The whole training took ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds.')

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  train_writer.close()
  sess.close()


if __name__ == '__main__':
  tf.app.run()
