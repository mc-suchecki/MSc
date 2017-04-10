import numpy
from matplotlib import pyplot
import sys

CAFFE_ROOT = '../../../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe


def visualize_weights(net, layer_name, padding=4, filename=''):
  # The parameters are a list of [weights, biases]
  data = numpy.copy(net.params[layer_name][0].data)
  # N is the total number of convolutions
  N = data.shape[0] * data.shape[1]
  # Ensure the resulting image is square
  filters_per_row = int(np.ceil(np.sqrt(N)))
  # Assume the filters are square
  filter_size = data.shape[2]
  # Size of the result image including padding
  result_size = filters_per_row * (filter_size + padding) - padding
  # Initialize result image to all zeros
  result = numpy.zeros((result_size, result_size))

  # Tile the filters into the result image
  filter_x = 0
  filter_y = 0
  for n in range(data.shape[0]):
    for c in range(data.shape[1]):
      if filter_x == filters_per_row:
        filter_y += 1
        filter_x = 0
      for i in range(filter_size):
        for j in range(filter_size):
          result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = data[n, c, i, j]
      filter_x += 1

  # Normalize image to 0-1
  min = result.min()
  max = result.max()
  result = (result - min) / (max - min)

  # Plot figure
  pyplot.figure(figsize=(10, 10))
  pyplot.axis('off')
  pyplot.imshow(result, cmap='gray', interpolation='nearest')

  # Save plot if filename is set
  if filename != '':
    pyplot.savefig(filename, bbox_inches='tight', pad_inches=0)

  # plt.show()


# load the model
net = caffe.Net('./model/alexnet/train_val.prototxt',
                './results/alexnet_model_iter_50000.caffemodel', caffe.TEST)

visualize_weights(net, 'conv1', filename='./results/alexnet/conv1.png')
visualize_weights(net, 'conv2', filename='./results/alexnet/conv2.png')
visualize_weights(net, 'conv3', filename='./results/alexnet/conv3.png')
visualize_weights(net, 'conv4', filename='./results/alexnet/conv4.png')
visualize_weights(net, 'conv5', filename='./results/alexnet/conv5.png')
