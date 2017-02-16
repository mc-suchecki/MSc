import os
import sys
import numpy
import matplotlib.pyplot

caffe_root = '../../../caffe/'
sys.path.insert(0, caffe_root + 'python/')
import caffe

# if os.path.isfile(caffe_root + 'models/finetune_flickr_style/finetune_flickr_style.caffemodel'):
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
  print('Caffe model found.')
else:
  print('Caffe model not found.')

caffe.set_device(0)
caffe.set_mode_gpu()

# model_def = caffe_root + 'models/finetune_flickr_style/deploy.prototxt'
# model_weights = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style.caffemodel'
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = numpy.load('../data/train/mean.npy')
mu = numpy.swapaxes(mu, 0, 2)
# mu2 = numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('Mean-subtracted values:', list(zip('BGR', mu)))

# create a Caffe transformer to process the image properly before inputting it to CNN
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# set the size of the input
net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 240x180

# load the image TODO put that in a loop later
image = caffe.io.load_image('../data/train/7225053870.jpg')
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()

# interpret the output
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print('Predicted class is:', output_prob.argmax())
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt' # load ImageNet labels
labels = numpy.loadtxt(labels_file, str, delimiter='\t')
print('Output label:', labels[output_prob.argmax()])
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print('Probabilities and labels:')
print(list(zip(output_prob[top_inds], labels[top_inds])))

# show the image
matplotlib.pyplot.imshow(image)
matplotlib.pyplot.show()
