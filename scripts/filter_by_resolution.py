"""Selects photos with given resolution and copies them to the desired folder, dividing into three sets."""
from PIL import Image
from progress.bar import Bar
from shutil import copyfile

# TODO modify the script in order to straighten the distribution of classes

# settings
SOURCE_DIRECTORY = '../data/'
PHOTOS_LIST = SOURCE_DIRECTORY + 'list.txt'
TRAINING_DIRECTORY = '../data/train/'
VALIDATION_DIRECTORY = '../data/validation/'
TEST_DIRECTORY = '../data/test/'
DESIRED_WIDTH = 240
DESIRED_HEIGHT = 180


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


with open(PHOTOS_LIST) as photos_list:
  number_of_photos = sum(1 for line in photos_list)

photos_copied = 0
with open(PHOTOS_LIST) as photos_list:
  progress_bar = Bar('Copying photos...', max=number_of_photos)
  for count, line in enumerate(photos_list, 1):
    progress_bar.next()
    photo_id = line.split(',')[0]
    file_name = photo_id + '.jpg'
    width, height = get_photo_dimensions(file_name)
    if width == DESIRED_WIDTH and height == DESIRED_HEIGHT:
      photos_copied += 1
      copyfile(SOURCE_DIRECTORY + file_name, get_destination_directory(count) + file_name)
      with open(get_destination_directory(count) + 'list.txt', "a") as destination_list_file:
        destination_list_file.write(line)

progress_bar.finish()
print('Copied ' + str(photos_copied) + ' photos.')
