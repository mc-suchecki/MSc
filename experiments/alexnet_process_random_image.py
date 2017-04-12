import os
import sys
import numpy
import random
import matplotlib.pyplot

CAFFE_ROOT = '../../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe

# settings
PHOTOS_LOCATION = '/media/p307k07/ssd/opt/msc/data/test/'
PHOTOS_LIST_LOCATION = PHOTOS_LOCATION + 'list.txt'
MEAN_FILE_LOCATION = PHOTOS_LOCATION + 'mean.npy'

if os.path.isfile('./results/alexnet_model_iter_450000.caffemodel'):
  print('Caffe model found.')
else:
  print('Caffe model not found.')
  quit()

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = CAFFE_ROOT + 'model/alexnet/deploy.prototxt'
model_weights = './results/alexnet_model_iter_450000.caffemodel'

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
net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          240, 159)  # image size is 240x159

# load the image
print('Getting a random photo from dataset...')
photos_list = open(PHOTOS_LIST_LOCATION, 'r').read().splitlines()
random_photo_location = PHOTOS_LOCATION + random.choice(photos_list).split(',')[0] + '.jpg'
print('Loading photo...')
image = caffe.io.load_image(random_photo_location)
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# perform classification
print('Classifying...')
output = net.forward()

# interpret the output
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print('AlexNet output for {} : {}'.format(random_photo_location, output_prob))

# show the image
# matplotlib.pyplot.imshow(image)
# matplotlib.pyplot.show()
