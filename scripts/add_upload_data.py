import flickrapi
import pyprind
import sys
import time
import requests

# settings
SOURCE_DIRECTORY = '../data/'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list.txt'
NEW_PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'list2.txt'


def get_photo_upload_date(photo_id):
  for _ in range(2):
    try:
      info_result = flickr.photos.getInfo(api_key=api_key, photo_id=photo_id, format='parsed-json')
      return info_result['photo']['dateuploaded']
    except (requests.exceptions.ConnectionError, KeyError, flickrapi.FlickrError):
      print('Getting upload date for photo {} failed! retrying...'.format(photo_id))
      time.sleep(2)
      continue
  return 0

# read API key and secret
with open('api_key.txt') as file:
  api_key = file.read()
with open('api_secret.txt') as file:
  api_secret = file.read()

# connect to Flickr
print('Connecting to Flickr...')
flickr = flickrapi.FlickrAPI(api_key, api_secret)

print('Reading photos list file...')
new_photo_list = []
with open(PHOTOS_LIST_FILE) as photos_list_file:
  photos_list = photos_list_file.readlines()
  print('Downloading new data for each photo and saving...')
  bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  with open(NEW_PHOTOS_LIST_FILE, 'w') as new_photos_list_file:
    for photo_metadata in photos_list:
      csv_list = photo_metadata.split(',')
      if len(csv_list) < 6:
        bar.update()
        photo_upload_date = get_photo_upload_date(int(csv_list[0]))
        if photo_upload_date == 0:
          continue
        csv_list[4] = csv_list[4].rstrip()
        csv_list.append(str(photo_upload_date) + '\n')
      new_photos_list_file.write(','.join(csv_list))

print('Done.')
