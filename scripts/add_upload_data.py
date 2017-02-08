import flickrapi
import pyprind
import sys
import time
import requests

# settings
SOURCE_DIRECTORY = '../data/'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list2.txt'
NEW_PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list3.txt'
ITERATIONS_LIMIT = 100000


def get_photo_upload_date(photo_id):
  while True:
    try:
      info_result = flickr.photos.getInfo(api_key=api_key, photo_id=photo_id, format='parsed-json')
      return info_result['photo']['dateuploaded']
    except (requests.exceptions.ConnectionError, KeyError, flickrapi.FlickrError):
      print('Getting upload date for photo #' + photo_id + ' failed! retrying...')
      time.sleep(2)
      continue

# read API key and secret
with open('api_key.txt') as file:
  api_key = file.read()
with open('api_secret.txt') as file:
  api_secret = file.read()

# connect to Flickr
print('Connecting to Flickr...')
flickr = flickrapi.FlickrAPI(api_key, api_secret)

print('Reading photos list file...')
iteration = 0
new_photo_list = []
with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()
  print('Downloading new data for each photo...')
  bar = pyprind.ProgBar(ITERATIONS_LIMIT, stream=sys.stdout, width=100)
  for photo_metadata in photos_list:
    csv_list = photo_metadata.split(',')
    if len(csv_list) < 6:
      bar.update()
      iteration += 1
      photo_upload_date = get_photo_upload_date(int(csv_list[0]))
      csv_list[4] = csv_list[4].rstrip()
      csv_list.append(str(photo_upload_date) + '\n')
    new_photo_list.append(','.join(csv_list))
    if iteration >= ITERATIONS_LIMIT:
      break
  updated_entries = len(new_photo_list)
  entries_left = len(photos_list) - updated_entries
  new_photo_list += photos_list[-entries_left:]

print('Writing new list to a file...')
with open(NEW_PHOTOS_LIST_FILE, 'w') as new_photos_list_file:
  new_photos_list_file.writelines(new_photo_list)

print('Done.')
