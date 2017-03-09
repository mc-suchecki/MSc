import numpy
import sys

CAFFE_ROOT = '../../../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe

# load mean file
mean = numpy.load('../data/train/mean.npy')

# convert to blob
blob = caffe.proto.caffe_pb2.BlobProto()
print('Original shape: ' + str(mean.shape))
mean = numpy.swapaxes(mean, 0, 2)  # first axis should be a channel
mean = numpy.swapaxes(mean, 1, 2)  # second axis should be height, third width
print('Corrected shape: ' + str(mean.shape))
blob.channels, blob.height, blob.width = mean.shape
print('{} channels, {} width, {} height.'.format(blob.channels, blob.width, blob.height))
blob.data.extend(mean.astype(float).flat)

# save
binaryproto_file = open('../data/train/mean.binaryproto', 'wb')
binaryproto_file.write(blob.SerializeToString())
binaryproto_file.close()
