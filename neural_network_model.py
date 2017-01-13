import tensorflow as tf


class NeuralNetworkModel(object):
  """ Provides helpers for easy construction of a neural network model. """

  @staticmethod
  def _weight_variable(shape):
    """ Creates a weight variable with appropriate initialization. """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
    """ Create a bias variable with appropriate initialization. """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @staticmethod
  def _variable_summaries(var):
    """ Attach a lot of summaries to a Tensor (for TensorBoard visualization). """
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  @staticmethod
  def _fully_connected_layer(input_tensor, input_dim, output_dim, layer_name, activation=tf.nn.relu):
    """
    Reusable code for making a simple neural net layer. It does a matrix multiply, bias add, and then uses an
    activation function to get the output of a layer. It also sets up name scoping so that the resultant graph is easy
    to read, and adds a number of summary ops.
    """
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = NeuralNetworkModel._weight_variable([input_dim, output_dim])
        NeuralNetworkModel._variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = NeuralNetworkModel._bias_variable([output_dim])
        NeuralNetworkModel._variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = activation(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  def _convolution_layer(self, input_tensor, filter_edge_length, num_of_output_channels, name):
    with tf.variable_scope(name) as scope:
      # last value in input tensor corresponds to the number of input channels in this layer
      input_channels = input_tensor.get_shape()[-1].value
      kernel_shape = [filter_edge_length, filter_edge_length, input_channels, num_of_output_channels]

      kernel = self._create_variable('weights', shape=kernel_shape, stddev=0.05, weight_decay=0.0)
      strides = [1, 1, 1, 1]  # steps for moving the filter

      conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME')
      biases = tf.get_variable('biases', [num_of_output_channels], initializer=tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)

      return conv2

  @staticmethod
  def _create_variable(name, shape, stddev, weight_decay):
    """
    Creates a variable, initializes it and adds a proper weight decay for training purposes. `weight_decay`
    parameter is multiplied by L2Loss weight decay and added to a TensorFlow collection of losses which are used
    in the loss function.
    """
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

    if weight_decay is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)

    return var
