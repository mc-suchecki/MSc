"""Selects photos with given resolution and copies them to the desired folder, dividing into three sets."""
import datetime
from math import log
from shutil import copyfile
import pyprind
import sys

# settings
SOURCE_DIRECTORY = '../data/'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list2.txt'
TRAINING_DIRECTORY = '../data/train/'
VALIDATION_DIRECTORY = '../data/validation/'
TEST_DIRECTORY = '../data/test/'
DESIRED_WIDTH = 240
DESIRED_HEIGHT = 180
VIEWS_THRESHOLD = 0
AESTHETICS_SCORE_MEDIAN = -7.247927513443586  # calculated by other script
NORMALIZED_VIEWS_MEDIAN = -5.272835310139874  # calculated by other script
today_timestamp = int(datetime.date.today().strftime("%s"))


# helpers
def get_destination_directory(number):
  """Returns training dir for 3/4 cases, validation dir for 1/8 cases and test dir for 1/8 cases."""
  if number % 2 == 0 or number % 4 == 1:
    return TRAINING_DIRECTORY
  elif number % 8 == 7:
    return VALIDATION_DIRECTORY
  else:
    return TEST_DIRECTORY


def get_photo_score(photo_metadata: str):
  """ Returns photo 'score'. The higher the ratio, the better. """
  stars = int(photo_metadata.split(',')[1])
  views = int(photo_metadata.split(',')[2])
  upload_date_timestamp = int(photo_metadata.split(',')[5])
  days_since_upload = abs(today_timestamp - upload_date_timestamp)/60/60/24
  # score = log((stars + 1) / (views + 1), 2)
  score = log((views + 1)/days_since_upload, 2)
  return score


def is_photo_resolution_okay(photo_metadata: str):
  """ Returns True if photo has proper resolution (defined in the settings at the beginning of the script). """
  width = int(photo_metadata.split(',')[3])
  height = int(photo_metadata.split(',')[4])
  return width == DESIRED_WIDTH and height == DESIRED_HEIGHT


with open(PHOTOS_LIST_FILE) as photos_list_file:
  # filter the data
  photos_list = photos_list_file.readlines()
  print('{} photos available.'.format(len(photos_list)))
  print('Removing the photos with resolution other than {}x{} pixels...'.format(DESIRED_WIDTH, DESIRED_HEIGHT))
  photos_list = [photo for photo in photos_list if is_photo_resolution_okay(photo)]
  print('{} photos left.'.format(len(photos_list)))
  print('Removing the photos with less than {} views...'.format(VIEWS_THRESHOLD))
  photos_list = [photo for photo in photos_list if int(photo.split(',')[2]) >= VIEWS_THRESHOLD]
  print('{} photos left.'.format(len(photos_list)))

  # copy the photos
  print('Copying photos...')
  progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  photo_index = 0
  good_photos = 0
  bad_photos = 0
  for line in photos_list:
    photo_index += 1
    progress_bar.update()
    photo_data = line.split(',')
    photo_id = photo_data[0]
    photo_label = 0 if get_photo_score(line) < NORMALIZED_VIEWS_MEDIAN else 1
    good_photos += photo_label
    bad_photos += 0 if photo_label == 1 else 1
    photo_file_name = photo_id + '.jpg'
    copyfile(SOURCE_DIRECTORY + photo_file_name, get_destination_directory(photo_index) + photo_file_name)
    with open(get_destination_directory(photo_index) + 'list.txt', "a") as destination_list_file:
      destination_list_file.write(','.join([photo_id, str(photo_label)]) + '\n')

  print('Copied ' + str(len(photos_list)) + ' photos.')
  print(
    'There were {} photos classified as aesthetically pleasing and {} not pleasing.'.format(good_photos, bad_photos))
