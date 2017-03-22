""" Parses passed log files and displays plots for losses and accuracy. """
from matplotlib import pylab
import click
import math
import numpy
import re

# number of columns in the plot - every cell contains a graph concerning one training run
# NOTE: should be bigger than 1 and less than number of files!
COLUMNS = 2


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
  """
  Parses passed log files and displays plots for losses and accuracy.
  :param files: paths to Caffe log files to parse
  """
  pylab.style.use('ggplot')

  rows = math.ceil(len(files) / COLUMNS)
  figure, axes = pylab.subplots(rows, COLUMNS)
  figure.suptitle('AlexNet training results - hyper-parameters optimization', fontsize=14)

  for index, filename in enumerate(files):
    first_axis = axes[index // rows, index % rows]
    second_axis = first_axis.twinx()
    loss_iterations, losses, test_iterations, accuracies, test_losses, title = parse_log(filename)
    first_axis.set_title(title, fontsize=11)
    # first_axis.set_xlabel('Iteration number', fontsize=10)
    first_axis.set_ylabel('Loss', fontsize=10)
    second_axis.set_ylabel('Accuracy (%)', fontsize=10)
    training_set_loss, = first_axis.plot(loss_iterations, losses, color='c', label='Training set loss')
    test_set_loss, = first_axis.plot(test_iterations, test_losses, 'b', label='Test set loss')
    test_set_accuracy, = second_axis.plot(test_iterations, accuracies, 'r', label='Test set accuracy')
    pylab.legend(handles=[training_set_loss, test_set_loss, test_set_accuracy], loc='upper left', prop={'size': 8})
    first_axis.set_ylim([0, 2])
    second_axis.set_ylim([50, 75])
    first_axis.set_xlim([0, 150000])

  pylab.show()


def parse_log(filename):
  """
  Parses passed log file and extracts losses and accuracy.
  :param log_file: path to the Caffe log file to parse
  :return: training losses, test losses, accuracies and corresponding iteration numbers
  """
  with open(filename) as log_file:
    log = log_file.read()

  # training data
  training_pattern = r".* Iteration (?P<iter_num>\d+) .*, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
  training_losses = []
  training_iterations = []

  for result in re.findall(training_pattern, log):
    training_iterations.append(int(result[0]))
    training_losses.append(float(result[1]))

  training_iterations = numpy.array(training_iterations)
  training_losses = numpy.array(training_losses)

  # test data
  test_pattern = r".* Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*.*accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\n.*loss = (?P<loss>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
  accuracies = []
  test_losses = []
  test_iterations = []

  for result in re.findall(test_pattern, log):
    iteration = int(result[0])
    accuracy = float(result[1]) * 100
    loss = float(result[5])
    test_iterations.append(iteration)
    accuracies.append(accuracy)
    test_losses.append(loss)

  test_iterations = numpy.array(test_iterations)
  accuracies = numpy.array(accuracies)
  test_losses = numpy.array(test_losses)

  # get the hyper-parameters to create the title for given plot
  base_learning_rate = re.findall(r"base_lr: (.*)", log)[0]
  momentum = re.findall(r"momentum: (.*)", log)[0]
  weight_decay = re.findall(r"weight_decay: (.*)", log)[0]
  dropout_ratio = re.findall(r"dropout_ratio: (.*)", log)
  title = 'Training ran on: ' + filename.split('/')[-1].split('_')[0] + '. ' 'Best accuracy: ' + \
          str(numpy.max(accuracies)) + '.\n' + 'Learning rate: ' + base_learning_rate + ', momentum: ' \
          + momentum + ', weight decay: ' + weight_decay + ', dropout: ' + dropout_ratio[0] + ' (FC6) / ' + \
          dropout_ratio[1] + ' (FC7).'

  return training_iterations, training_losses, test_iterations, accuracies, test_losses, title


if __name__ == '__main__':
  main()
