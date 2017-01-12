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
  def _fully_connected_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
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
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations
