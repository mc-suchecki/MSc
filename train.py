"""Trains neural network to assess the quality of photos."""
import datetime
import os

import tensorflow as tf

# TODO split this file
# TODO add TensorBoard

# we select horizontal photos with 3:2 aspect ratio as the most common type
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 159
NUMBER_OF_CHANNELS = 3
TRAINING_EXAMPLE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_CHANNELS
SOURCE_DIR = './data/240x159'
TRAINING_IMAGES_DIR = SOURCE_DIR + '/train/'               # 2100 images
VALIDATION_IMAGES_DIR = SOURCE_DIR + '/cross_validation/'  # 648 images
TEST_IMAGES_DIR = SOURCE_DIR + '/test/'                    # 900 images

BATCH_SIZE = 100
TRAINING_SET_SIZE = 2100
VALIDATION_SET_SIZE = 648
TEST_SET_SIZE = 900


def load_data_set(source_dir, set_size):
    # get list of photos
    filenames = []
    for filename in os.listdir(source_dir):
        filenames.append(filename)

    # get queue of photos
    filename_queue = tf.train.string_input_producer(filenames)
    photo_reader = tf.WholeFileReader()

    # read and convert photos
    filenames, contents = photo_reader.read(filename_queue)
    images = tf.image.decode_jpeg(contents, channels=NUMBER_OF_CHANNELS)
    images = tf.cast(images, tf.float32)
    images = tf.image.resize_images(images, [IMAGE_WIDTH, IMAGE_HEIGHT])
    images = tf.reshape(images, [-1])

    # read the label from photo filename
    filename_reader = tf.IdentityReader()
    filenames = filename_reader.read(filename_queue)
    # first number in filename is number of stars of a photo
    splitted_filenames = tf.string_split(filenames, '_')
    splitted_filenames = tf.sparse_tensor_to_dense(splitted_filenames, default_value=' ')
    stars_strings = splitted_filenames
    #stars_strings = tf.slice(splitted_filenames, [0, 0], [1, set_size])
    #stars_numbers = tf.string_to_number(stars_strings, tf.int32)
    # classify the photos by number of stars, with range of 5
    #labels = tf.floordiv(stars_numbers, tf.constant(5, tf.int32))

    return tf.train.batch([images, stars_strings], batch_size=BATCH_SIZE)


def main(_):
    # start measuring time
    start_time = datetime.datetime.now()

    # load the training set
    image_batch, label_batch = load_data_set(TRAINING_IMAGES_DIR, TRAINING_SET_SIZE)

    # create the model
    # x = tf.placeholder(tf.float32, [None, 240*159*3])       # input tensor
    x = image_batch
    W = tf.Variable(tf.zeros([240*159*3, 10]))              # weights tensor
    b = tf.Variable(tf.zeros([10]))                         # bias
    y = tf.matmul(x, W) + b
    # y_ = tf.placeholder(tf.float32, [None, 10])
    y_ = label_batch

    print(x.get_shape())
    print(W.get_shape())
    print(b.get_shape())
    print(y.get_shape())
    print(y_.get_shape())

    # define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # do the training
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    for i in range(TRAINING_SET_SIZE // BATCH_SIZE):
        print('Loading next ' + str(BATCH_SIZE) + ' photos...')
        print('Training batch ' + str(i) + '/' + str(TRAINING_SET_SIZE // BATCH_SIZE) + '...')
        sess.run(train_step)

    # test trained model
    print('Done! The whole training took ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds.')
    print('Evaluating the resulting model accuracy...')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    images, labels = load_data_set(TEST_IMAGES_DIR, TEST_SET_SIZE)
    print(sess.run(accuracy, feed_dict={x: images, y_: labels}))

if __name__ == '__main__':
    tf.app.run()
