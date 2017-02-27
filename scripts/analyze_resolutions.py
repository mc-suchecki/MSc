"""Displays a resolution breakdown - how many photos have a given resolution."""
from sortedcontainers import SortedDict
import pylab
import pyprind
import sys

# settings
SOURCE_DIRECTORY = '/media/p307k07/hdd/MSc/data/'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list.txt'
THRESHOLD = 10000

print('Calculating statistics...')
occurrences = SortedDict()
with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()
  progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  for photo_metadata in photos_list:
    width = int(photo_metadata.split(',')[3])
    height = int(photo_metadata.split(',')[4])
    # round the numbers to nearest 20s (we will crop the photos to have more for training)
    # width = int(pylab.math.ceil(width / 20.0)) * 20
    # height = int(pylab.math.ceil(height / 20.0)) * 20
    resolution_string = str(width) + 'x' + str(height)
    if resolution_string in occurrences:
      occurrences[resolution_string] += 1
    else:
      occurrences[resolution_string] = 0
    progress_bar.update()

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
