import flickrapi
import pyprind
import sys
import time
import requests

# settings
SOURCE_DIRECTORY = './'
PHOTOS_LIST_FILE = SOURCE_DIRECTORY + 'caffe_full_test_list.txt'
TAGS = ['landscape', 'portrait', 'architecture', 'macro', 'wildlife', 'blackandwhite']


def get_destination_file_for_photo(photo_id):
  for _ in range(5):
    try:
      info_result = flickr.photos.getInfo(api_key=api_key, photo_id=photo_id, format='parsed-json')
      tags = info_result['photo']['tags']['tag']
      for tag in tags:
        if tag['raw'] in TAGS:
          return tag['raw'] + '_photos_list.txt'
      return ''
    except (requests.exceptions.ConnectionError, KeyError, flickrapi.FlickrError):
      print('Getting tags for photo {} failed! retrying...'.format(photo_id))
      time.sleep(1)
      continue
  return ''


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
  print('Downloading tags data for each photo and dividing the dataset...')
  bar = pyprind.ProgBar(len(photos_list), stream=sys.stdout, width=100)
  for line in photos_list:
    bar.update()
    file_path = line.split(' ')[0]
    file_id = file_path.split('/')[-1].split('.')[0]
    destination_file_name = get_destination_file_for_photo(int(file_id))
    print(destination_file_name)
    if destination_file_name != '':
      with open('./' + destination_file_name, 'a') as new_photos_list_file:
        new_photos_list_file.write(line)

print('Done.')
