"""Selects photos with given resolution and copies them to the desired folder, dividing into three sets."""
from math import log
from progress.bar import Bar
from shutil import copyfile

# settings
SOURCE_DIRECTORY = '../data/'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list.txt'
TRAINING_DIRECTORY = '../data/train/'
VALIDATION_DIRECTORY = '../data/validation/'
TEST_DIRECTORY = '../data/test/'
DESIRED_WIDTH = 240
DESIRED_HEIGHT = 180
VIEWS_THRESHOLD = 100
AESTHETICS_SCORE_MEDIAN = -7.344295907915817  # calculated by other script


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
  score = log((stars + 1) / (views + 1), 2)
  return score


def is_photo_resolution_okay(photo_metadata: str):
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
  progress_bar = Bar('Copying photos...', max=len(photos_list))
  photo_index = 0
  good_photos = 0
  bad_photos = 0
  for line in photos_list:
    photo_index += 1
    progress_bar.next()
    photo_data = line.split(',')
    photo_id = photo_data[0]
    photo_label = 0 if get_photo_score(line) < AESTHETICS_SCORE_MEDIAN else 1
    good_photos += photo_label
    bad_photos += 0 if photo_label == 1 else 1
    photo_file_name = photo_id + '.jpg'
    copyfile(SOURCE_DIRECTORY + photo_file_name, get_destination_directory(photo_index) + photo_file_name)
    with open(get_destination_directory(photo_index) + 'list.txt', "a") as destination_list_file:
      destination_list_file.write(','.join([photo_id, str(photo_label)]) + '\n')

  progress_bar.finish()
  print('Copied ' + str(len(photos_list)) + ' photos.')
  print(
    'There were {} photos classified as aesthetically pleasing and {} not pleasing.'.format(good_photos, bad_photos))
