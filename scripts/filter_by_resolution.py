"""Selects photos with given resolution and copies them to the desired folder, dividing into three sets."""
import datetime
import itertools
import pyprind
import sys
import os
from math import log
from shutil import copyfile

# settings
SOURCE_DIRECTORY = '/media/p307k07/hdd/MSc/data/'
DESTINATION_DIRECTORY = '/media/p307k07/ssd/opt/msc/data'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list.txt'
TRAINING_DIRECTORY = DESTINATION_DIRECTORY + '/train/'
VALIDATION_DIRECTORY = DESTINATION_DIRECTORY + '/validation/'
TEST_DIRECTORY = DESTINATION_DIRECTORY + '/test/'
DESIRED_WIDTH = 240
DESIRED_HEIGHT = 159
VIEWS_THRESHOLD = 0
DESIRED_PERCENTAGE_OF_HIGH_QUALITY_PHOTOS = 20
DESIRED_PERCENTAGE_OF_LOW_QUALITY_PHOTOS = 20
NORMALIZED_VIEWS_MEDIAN = -3.4784655369370743  # calculated by other script
today_timestamp = int(datetime.date.today().strftime("%s"))


# helpers
def get_destination_directory(number):
  """Returns training dir for 3/4 cases, validation dir for 1/8 cases and test dir for 1/8 cases."""
  # we divide by 16 to ensure that validation and test datasets receive even number of photos
  # we need the above because photos in the list are cycling like (good, bad, good, bad, ...)
  remainder = number % 16
  if 0 <= remainder <= 11:
    return TRAINING_DIRECTORY
  elif remainder == 12 or remainder == 13:
    return VALIDATION_DIRECTORY
  else:
    return TEST_DIRECTORY


def get_photo_score(photo_metadata: str):
  """ Returns photo 'score'. The higher the returned number, the better the photo. """
  views = int(photo_metadata.split(',')[2])
  upload_date_timestamp = int(photo_metadata.split(',')[5])
  days_since_upload = abs(today_timestamp - upload_date_timestamp)/60/60/24
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
  print('Sorting the photos by score...')
  photos_list.sort(key=get_photo_score, reverse=True)
  print('Getting {}% best photos by score...'.format(DESIRED_PERCENTAGE_OF_HIGH_QUALITY_PHOTOS))
  number_of_best_photos = round(len(photos_list) * DESIRED_PERCENTAGE_OF_HIGH_QUALITY_PHOTOS / 100)
  best_photos_list = photos_list[:number_of_best_photos]
  print('Got {} best photos.'.format(len(best_photos_list)))
  print('Getting {}% worst photos by score...'.format(DESIRED_PERCENTAGE_OF_LOW_QUALITY_PHOTOS))
  number_of_worst_photos = round(len(photos_list) * DESIRED_PERCENTAGE_OF_LOW_QUALITY_PHOTOS / 100)
  worst_photos_list = (photos_list[-number_of_worst_photos:])[::-1]
  print('Got {} worst photos.'.format(len(worst_photos_list)))

  # copy the photos
  print('Copying photos...')
  progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  photo_index = 0
  good_photos = 0
  bad_photos = 0
  iterators = [iter(best_photos_list), iter(worst_photos_list)]
  for iterator in itertools.cycle(iterators):
    line = next(iterator)
    photo_index += 1
    photo_data = line.split(',')
    photo_id = photo_data[0]
    # photo_label = 0 if get_photo_score(line) < NORMALIZED_VIEWS_MEDIAN else 1
    photo_label = get_photo_score(line)
    good_photos += photo_label
    bad_photos += 0 if photo_label == 1 else 1
    photo_file_name = photo_id + '.jpg'
    path = get_destination_directory(photo_index)
    if not os.path.isfile(path + photo_file_name):
      copyfile(SOURCE_DIRECTORY + photo_file_name, path + photo_file_name)
    with open(path + 'list-regression.txt', 'a') as destination_list_file:
      destination_list_file.write(' '.join([path + photo_id + '.jpg', str(photo_label)]) + '\n')
    progress_bar.update()

  print('Copied ' + str(len(photos_list)) + ' photos.')
  print(
    'There were {} photos classified as aesthetically pleasing and {} not pleasing.'.format(good_photos, bad_photos))
