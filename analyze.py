from PIL import Image
from sortedcontainers import SortedDict
import os
import pylab

# settings
DIRECTORY = './data/'
VALID_EXTENSIONS = ('.JPG', '.jpg', '.jpeg')
THRESHOLD = 100


def is_photo(file_name):
  return file_name.endswith(VALID_EXTENSIONS)


def get_photo_dimensions(photo_path):
  image = Image.open(DIRECTORY + photo_path)
  image.verify()
  return image.size


print('Calculating statistics...')
occurrences = SortedDict()
for subdir, dirs, files in os.walk(DIRECTORY):
  for file in files:
    if is_photo(file):
      width, height = get_photo_dimensions(file)
      # round the numbers to 10s
      width = int(pylab.math.ceil(width / 20.0)) * 20
      height = int(pylab.math.ceil(height / 20.0)) * 20
      resolution_string = str(width) + 'x' + str(height)
      if resolution_string in occurrences:
        occurrences[resolution_string] += 1
      else:
        occurrences[resolution_string] = 0

print('Removing values lower than threshold...')
for key, value in list(occurrences.items()):
  if value < THRESHOLD:
    del occurrences[key]
print('Done!')


# plot the graph
position = pylab.arange(len(occurrences)) + .5
pylab.barh(position, occurrences.values(), align='center', color='#075AAD')
pylab.yticks(position, occurrences.keys())
pylab.xlabel('Occurrences')
pylab.ylabel('Resolution')
pylab.title('Resolutions breakdown')
pylab.grid(True)
pylab.show()
