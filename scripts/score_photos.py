import os
import sys

import pyprind

# settings
SCORES_TXT = '../data/test/list_scores.txt'
PHOTOS_LOCATION = '/media/p307k07/ssd/opt/msc/data/test/'
PHOTOS_LIST_FILE = PHOTOS_LOCATION + 'list.txt'
MODEL_WEIGHTS_LOCATION = '../experiments/results/alexnet_model_iter_450000.caffemodel'
MODEL_DEFINITION_LOCATION = '../experiments/model/alexnet/deploy.prototxt'
MEAN_VALUE_BLUE = 296
MEAN_VALUE_GREEN = 103
MEAN_VALUE_RED = 108


# checks
def check_file(path, message):
  if os.path.isfile(path):
    print(message + ' found.')
  else:
    print(message + ' not found! Exiting...')
    quit()


check_file(MODEL_WEIGHTS_LOCATION, 'Caffe model weights')
check_file(MODEL_DEFINITION_LOCATION, 'Caffe model definition')


# does inference in Caffe and returns the photo score
def get_photo_score(photo_path):
  photo = caffe.io.load_image(photo_path)
  transformed_image = transformer.preprocess('data', photo)
  net.blobs['data'].data[...] = transformed_image
  output = net.forward()
  score = output['prob'][0]  # the output probability vector for the first image in the batch
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

# process the photos
print('Scoring photos...')
with open(PHOTOS_LIST_FILE) as photos_list_file:
  with open(SCORES_TXT, 'a') as destination_list_file:
    photos_list = photos_list_file.readlines()
    progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)

    for line in photos_list:
      photo_metadata_list = line.split(',')
      photo_id = str(photo_metadata_list[0])
      photo_label = int(photo_metadata_list[1])
      score = get_photo_score(PHOTOS_LOCATION + photo_id + '.jpg')
      destination_list_file.write(' '.join([photo_id, str(score)]) + '\n')
      progress_bar.update()
