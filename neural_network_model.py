import tensorflow as tf


class NeuralNetworkModel(object):
  """ Provides helpers for easy construction of a neural network model. """

  def generate_network_model(self, input_tensor: tf.Tensor, reuse_variables=False) -> tf.Tensor:
    """ TODO comment this. """
    with tf.variable_scope('neural_network', reuse=reuse_variables):
      conv1 = self._convolutional_layer('1_convolution', input_tensor, 5, 64, reuse_variables)
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='2_pooling')
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='3_normalization')

      conv2 = self._convolutional_layer('4_convolution', norm1, 5, 64, reuse_variables)
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='5_normalization')
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='6_pooling')

      fc1 = self._fully_connected_layer('7_fully_connected', pool2, 384, reuse_variables)
      fc2 = self._fully_connected_layer('8_fully_connected', fc1, 192, reuse_variables)

      output = self._output_layer(fc2, reuse_variables)
      output = tf.Print(output, [output], message="Last layer activations: ", summarize=10)

    return output

  def _convolutional_layer(self, layer_name, input_tensor, filter_size, num_of_output_channels, reuse_variables):
    """ TODO comment this. """
    with tf.variable_scope(layer_name, reuse=reuse_variables) as scope:
      # last value in input tensor corresponds to the number of input channels in this layer
      input_channels = input_tensor.get_shape()[-1].value
      kernel_shape = [filter_size, filter_size, input_channels, num_of_output_channels]

      kernel = self._create_variable('weights', shape=kernel_shape, stddev=0.05, weight_decay=0.0)
      strides = [1, 1, 1, 1]  # steps for moving the filter

      conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME')
      biases = tf.get_variable('biases', [num_of_output_channels], initializer=tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      output = tf.nn.relu(pre_activation, name=scope.name)
      self._generate_summary(output)
      return output

  def _fully_connected_layer(self, layer_name, input_tensor, num_of_outputs, reuse_variables):
    """ TODO comment this. """
    with tf.variable_scope(layer_name, reuse=reuse_variables) as scope:
      batch_size = input_tensor.get_shape()[0].value
      # flatten the tensor so that the output can be calculated by simple matrix multiplication
      reshape = tf.reshape(input_tensor, [batch_size, -1])
      num_of_input_values = reshape.get_shape()[1].value
      weights = self._create_variable('weights', shape=[num_of_input_values, num_of_outputs],
                                      stddev=0.04, weight_decay=0.004)
      biases = tf.get_variable('biases', [num_of_outputs], initializer=tf.constant_initializer(0.1))
      output = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      self._generate_summary(output)
      return output

  def _output_layer(self, input_tensor, reuse_variables):
    """ TODO comment this. """
    with tf.variable_scope('output_layer', reuse=reuse_variables) as scope:
      inputs_length = input_tensor.get_shape()[-1].value
      weights = self._create_variable('weights', [inputs_length, 1], stddev=1 / inputs_length, weight_decay=0.0)
      # weights = tf.Print(weights, [weights], message="Output layer weights: ", summarize=192)
      biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
      # biases = tf.Print(biases, [biases], message="Output layer bias: ", summarize=1)
      output = tf.sigmoid(tf.add(tf.matmul(input_tensor, weights), biases, name=scope.name))
      self._generate_summary(output)
      return output

  @staticmethod
  def _create_variable(name, shape, stddev, weight_decay):
    """ Creates a TensorFlow variable, initializes it randomly and adds a weight decay for training purposes. """
    variable = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

    if weight_decay is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)

    return variable

  @staticmethod
  def _generate_summary(x):
    """ TODO comment this. """
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))
