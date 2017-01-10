"""Trains neural network to assess the quality of photos."""
import datetime
import tensorflow as tf

# TODO move loading of the dataset to a separate file
# TODO take batch size as a parameter
# TODO read set sizes from list files
# TODO re-think the computation graph and add namespaces
# TODO improve evaluation on the validation set - so far we only evaluate on 100 images
# TODO evaluate the model on the test set at the end
# TODO find a way to plot accuracy on the training set also?
# TODO save the model after training and write a script to process a given photo

# we select horizontal photos with 4:3 aspect ratio as the most common type
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 180
NUMBER_OF_CHANNELS = 3
TRAINING_EXAMPLE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_CHANNELS
SOURCE_DIR = './data/'
TRAINING_IMAGES_DIR = SOURCE_DIR + 'train/'
VALIDATION_IMAGES_DIR = SOURCE_DIR + 'validation/'
TEST_IMAGES_DIR = SOURCE_DIR + 'test/'
LIST_FILE_NAME = 'list.txt'
BATCH_SIZE = 100
TRAINING_SET_SIZE = 15873


def create_photo_and_label_batches(source_directory):
  # read the list of photo IDs and labels
  photos_list = open(source_directory + LIST_FILE_NAME, 'r')
  filenames_list = []
  labels_list = []
  # get lists of photo file names and labels
  for line in photos_list:
    filenames_list.append(source_directory + line.split(',')[0] + '.jpg')
    # so far the naive approach is to assess an photo as aesthetically pleasing if it has at least one star
    labels_list.append([bool(int(line.split(',')[1]))])        # TODO improve the assignment of the classes
  # convert the lists to tensors
  filenames = tf.convert_to_tensor(filenames_list, dtype=tf.string)
  labels = tf.convert_to_tensor(labels_list, dtype=tf.bool)
  # create queue with filenames and labels
  file_names_queue, labels_queue = tf.train.slice_input_producer([filenames, labels], shuffle=True)
  # convert filenames of photos to input vectors
  photos_queue = tf.read_file(file_names_queue)                                     # open the file with the photo
  photos_queue = tf.image.decode_jpeg(photos_queue, channels=NUMBER_OF_CHANNELS)    # convert file to numbers
  photos_queue.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_CHANNELS])           # tell TF the size of images
  photos_queue = tf.to_float(photos_queue)                                          # convert uint8 to float32
  photos_queue = tf.reshape(photos_queue, [-1])                                     # flatten the tensor (TODO remove)
  # slice the data into mini batches
  return tf.train.batch([photos_queue, labels_queue], batch_size=BATCH_SIZE)


def main(_):
  # start measuring time
  start_time = datetime.datetime.now()

  # load the datasets
  training_photo_batch, training_label_batch = create_photo_and_label_batches(TRAINING_IMAGES_DIR)
  validation_photo_batch, validation_label_batch = create_photo_and_label_batches(VALIDATION_IMAGES_DIR)
  # test_photo_batch, test_label_batch = create_photo_and_label_batches(TEST_IMAGES_DIR)

  # create the model
  x = training_photo_batch                                                        # input tensor (so far flattened)
  w = tf.Variable(tf.zeros([TRAINING_EXAMPLE_SIZE, 1], dtype=tf.float32))         # weights tensor
  b = tf.Variable(tf.zeros([1], dtype=tf.float32))                                # bias tensor (so far only 1 neuron)
  y_ = training_label_batch                                                       # labels tensor
  y = tf.sigmoid(tf.matmul(x, w) + b)                                             # predictions

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
  train_writer = tf.summary.FileWriter('/tmp/tf/train', sess.graph)
  validation_writer = tf.summary.FileWriter('/tmp/tf/validation', sess.graph)

  # do the training
  tf.initialize_all_variables().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  number_of_batches = TRAINING_SET_SIZE // BATCH_SIZE
  for i in range(number_of_batches):
    print('Training with batch ' + str(i) + '/' + str(number_of_batches) + ' (' + str(BATCH_SIZE) + ' photos)...')
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
