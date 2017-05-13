import io
import os
import sys
import zmq
import logging

from PIL import Image

# config
MODEL_WEIGHTS_LOCATION = '../experiments/results/alexnet_model_iter_450000.caffemodel'
MODEL_DEFINITION_LOCATION = '../experiments/model/alexnet/deploy.prototxt'
MEAN_VALUE_BLUE = 296
MEAN_VALUE_GREEN = 103
MEAN_VALUE_RED = 108

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# checks
def check_file(path, message):
  if os.path.isfile(path):
    logger.info(message + ' found.')
  else:
    logger.error(message + ' not found! Exiting...')
    quit()

check_file(MODEL_WEIGHTS_LOCATION, 'Caffe model weights')
check_file(MODEL_DEFINITION_LOCATION, 'Caffe model definition')


# does inference in Caffe and returns the photo score
def get_photo_score(photo):
  if photo.size[0] < photo.size[1]:
    logging.info("Image is vertical - rotating to horizontal...")
    photo = photo.rotate(90)

  logging.info("Resizing the image to 240x159 pixels...")
  photo.resize((240, 159), 3)

  logging.info("Loading image to Caffe...")
  photo.save("./temp.jpg")
  photo = caffe.io.load_image("./temp.jpg")
  os.remove("./temp.jpg")
  transformed_image = transformer.preprocess('data', photo)
  net.blobs['data'].data[...] = transformed_image

  logging.info("Doing the forward propagation...")
  output = net.forward()

  score = output['prob'][0]  # the output probability vector for the first image in the batch
  logging.info("Done. Photo score is {}%.".format(score))
  return score


# init Caffe
CAFFE_ROOT = '../../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(MODEL_DEFINITION_LOCATION, MODEL_WEIGHTS_LOCATION, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', [MEAN_VALUE_RED, MEAN_VALUE_GREEN, MEAN_VALUE_BLUE])  # subtract the mean
transformer.set_raw_scale('data', 255)  # rescale from [0, 255] to [0, 1]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          240, 159)  # image size is 240x159

# create context and specify the port
context = zmq.Context()
logger.info("Creating the socket using port 6666...")
socket = context.socket(zmq.REP)
socket.bind("tcp://*:6666")
logger.info("Socket successfully created!")
logger.info("Waiting for requests...")

# wait for requests and respond
while True:
  data = socket.recv()
  logger.info("Received a photo!")
  stream = io.BytesIO(data)
  image = Image.open(stream)
  logger.info("Photo resolution is {}x{} pixels.".format(image.size[0], image.size[1]))
  logger.info("Starting inference using AlexNet...")
  score = get_photo_score(image)
  socket.send_json({"score": score})
