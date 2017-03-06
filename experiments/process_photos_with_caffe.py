import os
import sys
import numpy
import pyprind

CAFFE_ROOT = '../../../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe

# settings
PHOTOS_LOCATION = '/media/p307k07/ssd/opt/msc/data/train/'
PHOTOS_LIST_LOCATION = PHOTOS_LOCATION + 'list.txt'
MEAN_FILE_LOCATION = PHOTOS_LOCATION + 'mean.npy'
OUTPUT_LOCATION = '../data/train/'
PHOTOS_LIMIT = 2**12

if os.path.isfile(CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
  print('Caffe model found.')
else:
  print('Caffe model not found.')

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy_modified.prototxt'
model_weights = CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# load the mean image from the dataset
mu = numpy.load(MEAN_FILE_LOCATION)
mu = numpy.swapaxes(mu, 0, 2)  # Caffe expects the mean to be in shape (CHANNELS x WIDTH x HEIGHT)
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('Mean-subtracted values:', list(zip('BGR', mu)))

# create a Caffe transformer to process the image properly before inputting it to CNN
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# set the size of the input
net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 240x180

# load the images, process them with Caffe and save the second-to-last layer output for each image
print('Loading photos...')
training_examples = []
training_labels = []
with open(PHOTOS_LIST_LOCATION) as photos_list_file:
  photos_list = photos_list_file.readlines()
  photos_list = photos_list[:PHOTOS_LIMIT]

  print('Computing Caffe output for {} photos...'.format(len(photos_list)))
  progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  for line in photos_list:
    progress_bar.update()
    photo_location = PHOTOS_LOCATION + line.split(',')[0] + '.jpg'
    photo_label = line.split(',')[1]
    image = caffe.io.load_image(photo_location)
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()

    # save the activations of last fully connected layer and appropriate labels
    training_examples.append(net.blobs['fc6'].data[0].tolist())
    training_labels.append(photo_label[0])

numpy.save(OUTPUT_LOCATION + 'X_fc6_cut.npy', training_examples)
numpy.save(OUTPUT_LOCATION + 'y_fc6_cut.npy', training_labels)
