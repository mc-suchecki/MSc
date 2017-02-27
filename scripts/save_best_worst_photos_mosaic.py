from PIL import Image

# settings
PHOTOS_LOCATION = '../data/validation/'
PHOTOS_LIST_FILE = PHOTOS_LOCATION + 'list.txt'
PHOTOS_WIDTH = 240
PHOTOS_HEIGHT = 159
MOSAIC_SIZE = 10

# init
number_of_photos_to_process = MOSAIC_SIZE * MOSAIC_SIZE * 2
best_mosaic = Image.new('RGB', (PHOTOS_WIDTH * MOSAIC_SIZE, PHOTOS_HEIGHT * MOSAIC_SIZE))
worst_mosaic = Image.new('RGB', (PHOTOS_WIDTH * MOSAIC_SIZE, PHOTOS_HEIGHT * MOSAIC_SIZE))
with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()
  best_photos_processed = 0
  worst_photos_processed = 0

  for line in photos_list:
    photo_metadata_list = line.split(',')
    photo_id = str(photo_metadata_list[0])
    photo_label = int(photo_metadata_list[1])
    photo = Image.open(PHOTOS_LOCATION + photo_id + '.jpg')

    if photo_label == 0:
      x = worst_photos_processed % MOSAIC_SIZE * PHOTOS_WIDTH
      y = worst_photos_processed // MOSAIC_SIZE * PHOTOS_HEIGHT
      worst_photos_processed += 1
      worst_mosaic.paste(photo, (x, y, x + PHOTOS_WIDTH, y + PHOTOS_HEIGHT))
    else:
      x = best_photos_processed % MOSAIC_SIZE * PHOTOS_WIDTH
      y = best_photos_processed // MOSAIC_SIZE * PHOTOS_HEIGHT
      best_photos_processed += 1
      best_mosaic.paste(photo, (x, y, x + PHOTOS_WIDTH, y + PHOTOS_HEIGHT))

    if best_photos_processed == number_of_photos_to_process and worst_photos_processed == number_of_photos_to_process:
      break

best_mosaic.save(PHOTOS_LOCATION + 'best_photos.jpg')
worst_mosaic.save(PHOTOS_LOCATION + 'worst_photos.jpg')
