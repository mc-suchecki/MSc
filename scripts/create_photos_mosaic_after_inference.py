import sys

import pyprind
from PIL import Image

# settings
PHOTOS_LOCATION = '/media/p307k07/ssd/opt/msc/data/test/'
PHOTOS_LIST_FILE = PHOTOS_LOCATION + 'list_scored.txt'
OUTPUT_LOCATION = '../data/test/'
PHOTOS_WIDTH = 240
PHOTOS_HEIGHT = 159
MOSAIC_SIZE = 10

# traverse the list file to get best and worst photos
print("Processing the photos...")
number_of_photos_to_process = MOSAIC_SIZE * MOSAIC_SIZE
with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()
  progress_bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  best_photos_list = []
  worst_photos_list = []

  for line in photos_list:
    photo_metadata_list = line.split(',')
    photo_id = str(photo_metadata_list[0])
    photo_score = int(photo_metadata_list[1])

    best_photos_list.append((photo_id, photo_score))
    worst_photos_list.append((photo_id, photo_score))
    best_photos_list.sort(key=lambda photo_data: photo_data[1], reverse=True)  # sort by score
    worst_photos_list.sort(key=lambda photo_data: photo_data[1])  # sort by score
    best_photos_list = best_photos_list[:number_of_photos_to_process]  # get first n elements
    worst_photos_list = worst_photos_list[:number_of_photos_to_process]  # get first n elements

    progress_bar.update()

print("Done! Got {} best and {} worst photos.".format(len(best_photos_list), len(worst_photos_list)))

# create the mosaics
print("Creating the mosaics...")
best_mosaic = Image.new('RGB', (PHOTOS_WIDTH * MOSAIC_SIZE, PHOTOS_HEIGHT * MOSAIC_SIZE))
worst_photos_processed = 0
best_photos_processed = 0
for photo_metadata in best_photos_list:
  photo = Image.open(PHOTOS_LOCATION + photo_metadata[1] + '.jpg')
  x = best_photos_processed % MOSAIC_SIZE * PHOTOS_WIDTH
  y = best_photos_processed // MOSAIC_SIZE * PHOTOS_HEIGHT
  best_photos_processed += 1
  best_mosaic.paste(photo, (x, y, x + PHOTOS_WIDTH, y + PHOTOS_HEIGHT))
best_mosaic.save(OUTPUT_LOCATION + 'best_photos_according_to_model.jpg')

worst_mosaic = Image.new('RGB', (PHOTOS_WIDTH * MOSAIC_SIZE, PHOTOS_HEIGHT * MOSAIC_SIZE))
for photo_metadata in worst_photos_list:
  photo = Image.open(PHOTOS_LOCATION + photo_metadata[1] + '.jpg')
  x = worst_photos_processed % MOSAIC_SIZE * PHOTOS_WIDTH
  y = worst_photos_processed // MOSAIC_SIZE * PHOTOS_HEIGHT
  worst_photos_processed += 1
  worst_mosaic.paste(photo, (x, y, x + PHOTOS_WIDTH, y + PHOTOS_HEIGHT))
worst_mosaic.save(OUTPUT_LOCATION + 'worst_photos_according_to_model.jpg')
