""" Parses passed log files and displays plots for losses and accuracy. """
import numpy as np
import re
import click
from matplotlib import pylab as plt


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
  """
  Parses passed log files and displays plots for losses and accuracy.
  :param files: paths to Caffe log files to parse
  """
  plt.style.use('ggplot')
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  fig.suptitle('AlexNet training with learning rate = 0.001 and fixed convolutional layers', fontsize=20)
  ax1.set_xlabel('Iteration number', fontsize=14)
  ax1.set_ylabel('Loss', fontsize=14)
  ax2.set_ylabel('Accuracy (%)', fontsize=14)

  for log_file in files:
    loss_iterations, losses, test_iterations, accuracies, test_losses = parse_log(log_file)
    training_set_loss, = ax1.plot(loss_iterations, losses, color='y', label='Training set loss')
    test_set_loss, = ax1.plot(test_iterations, test_losses, 'r', label='Test set loss')
    test_set_accuracy, = ax2.plot(test_iterations, accuracies, 'b', label='Test set accuracy')
    plt.legend(handles=[training_set_loss, test_set_loss, test_set_accuracy], loc='upper left')

  plt.show()


def parse_log(log_file):
  """
  Parses passed log file and extracts losses and accuracy.
  :param log_file: path to the Caffe log file to parse
  :return: training losses, test losses, accuracies and corresponding iteration numbers
  """
  with open(log_file) as log_file:
    log = log_file.read()

  # training data
  training_pattern = r".* Iteration (?P<iter_num>\d+) .*, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
  training_losses = []
  training_iterations = []

  for result in re.findall(training_pattern, log):
    training_iterations.append(int(result[0]))
    training_losses.append(float(result[1]))

  training_iterations = np.array(training_iterations)
  training_losses = np.array(training_losses)

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

  test_iterations = np.array(test_iterations)
  accuracies = np.array(accuracies)
  test_losses = np.array(test_losses)

  return training_iterations, training_losses, test_iterations, accuracies, test_losses


if __name__ == '__main__':
  main()
