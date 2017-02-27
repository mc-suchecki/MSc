from PIL import Image

# settings
PHOTOS_LOCATION = '../data/train/'
PHOTOS_LIST_FILE = PHOTOS_LOCATION + 'list.txt'

with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()

  for line in photos_list:
    photo_metadata_list = line.split(',')
    photo_id = str(photo_metadata_list[0])
    photo_label = int(photo_metadata_list[1])
    photo = Image.open(PHOTOS_LOCATION + photo_id + '.jpg')
    photo.show()
    input('Showing a {} photo. Press any key for the next one.'.format('bad' if photo_label == 0 else 'good'))
