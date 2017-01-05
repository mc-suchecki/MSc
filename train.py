"""Trains neural network to assess the quality of photos."""
import datetime
import tensorflow as tf

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
NUMBER_OF_EPOCHS = 1
TRAINING_SET_SIZE = 15873                                # TODO read this from file?


def create_photo_and_label_batches(source_directory):
  """TODO comment, split and throw to another file."""
  # read the list of photo IDs and labels
  photos_list = open(source_directory + LIST_FILE_NAME, 'r')
  filenames_list = []
  labels_list = []
  # get lists of photo file names and labels
  for line in photos_list:
    filenames_list.append(source_directory + line.split(',')[0] + '.jpg')
    # so far the naive approach is to asses an photo as aesthetically pleasing if it has at least one star
    labels_list.append([bool(int(line.split(',')[1]))])        # TODO improve the assignment of the classes
  # convert the lists to tensors
  filenames = tf.convert_to_tensor(filenames_list, dtype=tf.string)
  labels = tf.convert_to_tensor(labels_list, dtype=tf.bool)
  # create queue with filenames and labels
  file_names_queue, labels_queue = tf.train.slice_input_producer([filenames, labels], num_epochs=NUMBER_OF_EPOCHS,
                                                                 shuffle=True)
  # convert filenames of photos to input vectors
  photos_queue = tf.read_file(file_names_queue)                                     # open the file with the photo
  photos_queue = tf.image.decode_jpeg(photos_queue, channels=NUMBER_OF_CHANNELS)    # convert file to numbers
  photos_queue.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_CHANNELS])           # tell TF the size of images
  photos_queue = tf.to_float(photos_queue)                                          # convert uint8 to float32
  photos_queue = tf.reshape(photos_queue, [-1])                                     # flatten the tensor
  # slice the data into mini batches
  return tf.train.batch([photos_queue, labels_queue], batch_size=BATCH_SIZE)


def main(_):
  # start measuring time
  start_time = datetime.datetime.now()

  # load the training set
  training_photo_batch, training_label_batch = create_photo_and_label_batches(TRAINING_IMAGES_DIR)

  # create the model
  # x = tf.placeholder(tf.float32, [None, 240*159*3])       # input tensor
  x = training_photo_batch
  W = tf.Variable(tf.zeros([IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_CHANNELS, 1], dtype=tf.float32))  # weights tensor
  # b = tf.Variable(tf.zeros([1], dtype=tf.float32))  # bias

  print(x.get_shape())
  print(W.get_shape())
  # print(b.get_shape())
  # y_ = tf.placeholder(tf.float32, [None, 10])
  y_ = training_label_batch
  print(y_.get_shape())
  # y = tf.matmul(x, W) + b
  y = tf.matmul(x, W)
  print(y.get_shape())

  # define loss and optimizer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # initialize session
  sess = tf.InteractiveSession()

  # initialize TensorBoard
  tf.summary.image('Input images', x)
  summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/tmp/tf', sess.graph)

  # do the training
  tf.initialize_all_variables().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(TRAINING_SET_SIZE // BATCH_SIZE):
    print('Loading next ' + str(BATCH_SIZE) + ' photos...')
    print('Training batch ' + str(i) + '/' + str(TRAINING_SET_SIZE // BATCH_SIZE) + '...')
    sess.run(train_step)
    sess.run(summaries)

  # test trained model
  print('Done! The whole training took ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds.')
  # print('Evaluating the resulting model accuracy...')
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # test_photo_batch, test_label_batch = create_photo_and_label_batches(TEST_IMAGES_DIR)
  # test_label_batch = create_photo_and_label_batches(TEST_IMAGES_DIR + LIST_FILE_NAME)
  # print(sess.run(accuracy, feed_dict={x: test_photo_batch, y_: test_label_batch}))

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  train_writer.close()
  sess.close()


if __name__ == '__main__':
  tf.app.run()
