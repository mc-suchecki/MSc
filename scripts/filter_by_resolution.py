"""Selects photos with given resolution and copies them to the desired folder, dividing into three sets."""
from PIL import Image
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
DESIRED_NUMBER_OF_PHOTOS = 12000


# helpers
def get_photo_dimensions(file_name):
  """Returns photo resolution for a given filename."""
  image = Image.open(SOURCE_DIRECTORY + str(file_name))
  image.verify()
  return image.size


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
  # TODO so far just stars count seems better - check this!
  stars = int(photo_metadata.split(',')[1])
  views = int(photo_metadata.split(',')[2])
  return stars


def is_photo_resolution_okay(photo_metadata: str):
  width = int(photo_metadata.split(',')[3])
  height = int(photo_metadata.split(',')[4])
  return width == DESIRED_WIDTH and height == DESIRED_HEIGHT

photos_copied = int(0)
with open(PHOTOS_LIST_FILE) as photos_list_file:
  # filter the data
  photos_list = photos_list_file.readlines()
  print('Found {} photos available.'.format(len(photos_list)))
  print('Removing the photos with resolution other than {}x{} pixels...'.format(DESIRED_WIDTH, DESIRED_HEIGHT))
  photos_list = [photo for photo in photos_list if is_photo_resolution_okay(photo)]
  print('{} photos left.'.format(len(photos_list)))
  print('Removing the photos with less than 100 views...')
  photos_list = [photo for photo in photos_list if int(photo.split(',')[2]) >= 100]
  print('{} photos left.'.format(len(photos_list)))
  print('Sorting the photos by stars count...')
  photos_list.sort(key=get_photo_score)

  # copy the photos
  progress_bar = Bar('Copying photos...', max=DESIRED_NUMBER_OF_PHOTOS)
  last_worst_photo_index = 0
  last_best_photo_index = len(photos_list) - 1
  last_best_photo_stars = 999999999
  while (photos_copied < DESIRED_NUMBER_OF_PHOTOS) and (int(last_best_photo_stars) > 0):
    progress_bar.next()
    worst_photo = photos_list[last_worst_photo_index]
    best_photo = photos_list[last_best_photo_index]

    # copy the current best and worst photos
    best_photo_id = best_photo.split(',')[0]
    best_file_name = best_photo_id + '.jpg'
    copyfile(SOURCE_DIRECTORY + best_file_name, get_destination_directory(last_worst_photo_index) + best_file_name)
    with open(get_destination_directory(last_worst_photo_index) + 'list.txt', "a") as destination_list_file:
      destination_list_file.write(best_photo)
    worst_photo_id = worst_photo.split(',')[0]
    worst_file_name = worst_photo_id + '.jpg'
    copyfile(SOURCE_DIRECTORY + worst_file_name, get_destination_directory(last_worst_photo_index) + worst_file_name)
    with open(get_destination_directory(last_worst_photo_index) + 'list.txt', "a") as destination_list_file:
      destination_list_file.write(worst_photo)

    # save the number of stars for the last best photo (to prevent copying too much photos with 0 stars and skewed data)
    last_best_photo_stars = best_photo.split(',')[1]

    # move the indexes further
    last_worst_photo_index += 1
    last_best_photo_index -= 1
    photos_copied += 2

  progress_bar.finish()
  print('Copied ' + str(photos_copied) + ' photos.')
