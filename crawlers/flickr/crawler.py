"""Downloads all of the public photographs from Flickr."""
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from PIL import Image
import datetime
import flickrapi
import os
import requests
import time

# 'must have' TODOs
# TODO what to do about the aspect ratio?

# 'maybe' TODOs
# TODO think about different size
# TODO better way to avoid pictures not being photos

# settings
# DOWNLOAD_LOCATION = '/home/p307k07/Code/MSc/MSc/data/'        # laboratory
DOWNLOAD_LOCATION = '/home/mc/Code/Python/MSc/data/'  # ThinkPad
POOL_SIZE = 20  # number of subprocesses to spawn while downloading photos
PAGE_SIZE = 500  # Flickr allows maximum 500 photos per page
TAGS = ','.join(['landscape', 'portrait', 'architecture', 'macro', 'street', 'travel', 'nature', 'wildlife', 'night',
                 'blackandwhite'])  # we limit the results to only these tags to avoid documents etc
EXTRAS = ','.join(['views'])  # download also number of views
PHOTO_SIZE = 'm'  # resolution of the photos - 'm' means small, 240 on longest side


# functions
def get_flickr_photo_url(farm_id, server_id, photo_id, secret):
  return 'https://farm{}.staticflickr.com/{}/{}_{}_{}.jpg'.format(farm_id, server_id, photo_id, secret, PHOTO_SIZE)


def download_flickr_photo(photo_file_name, photo_url):
  while True:
    try:
      response = requests.get(photo_url)
      with open(photo_file_name, 'wb') as photo_file:
        photo_file.write(response.content)
    except requests.exceptions.ConnectionError:
      print('Downloading from ' + photo_url + ' failed!')
      print('Retrying...')
      time.sleep(2)
      continue
    break


def get_photo_favorites(photo_id):
  while True:
    try:
      fav_result = flickr.photos.getFavorites(api_key=api_key, photo_id=photo_id, format='parsed-json')
      return fav_result['photo']['total']
    except requests.exceptions.ConnectionError:
      print('Downloading stars for photo #' + photo_id + ' failed!')
      print('Retrying...')
      time.sleep(2)
      continue


def save_flickr_photo_to_disk(photo_info):
  photo_id = photo_info['id']
  file_name = DOWNLOAD_LOCATION + str(photo_id) + '.jpg'

  # check if photo was already downloaded
  if os.path.isfile(file_name):
    print('Skipping ' + file_name + '... (already downloaded)')
    return

  # download the photo
  url = get_flickr_photo_url(photo_info['farm'], photo_info['server'], photo_id, photo_info['secret'])
  print('Downloading from ' + url + '...')
  download_flickr_photo(file_name, url)

  # save the photo ID to a list along with additional data
  try:
    favorites = get_photo_favorites(photo_id)
    # getFavorites sometimes throws 'error 1: Photo not found' even though photo exists...
  except flickrapi.exceptions.FlickrError:
    return
  views = photo_info['views']
  with Image.open(file_name) as photo_file:
    width, height = photo_file.size
  with open(DOWNLOAD_LOCATION + 'list.txt', "a") as list_file:
    list_file.write(','.join([photo_id, favorites, str(views), str(width), str(height)]) + '\n')


# read API key and secret
with open('api_key.txt') as file:
  api_key = file.read()
with open('api_secret.txt') as file:
  api_secret = file.read()

# connect to Flickr
flickr = flickrapi.FlickrAPI(api_key, api_secret)

# Flickr returns only 4000 unique results, so we need to do multiple queries, here we go iterating by month
min_upload_date = datetime.datetime(2000, 1, 1)
while min_upload_date <= datetime.datetime.now():
  max_upload_date = min_upload_date + relativedelta(months=1)
  result = flickr.photos.search(api_key=api_key, min_upload_date=min_upload_date, max_upload_date=max_upload_date,
                                media='photos', content_type=1, tags=TAGS, per_page=PAGE_SIZE, tag_mode='any',
                                extras=EXTRAS, format='parsed-json')

  number_of_photos = len(result['photos']['photo'])
  print('Downloading ' + str(number_of_photos) + ' photos from ' + min_upload_date.strftime('%m/%Y') + '...')

  # open a subprocess for every photo to speed up downloading
  with Pool(POOL_SIZE) as pool:
    pool.map(save_flickr_photo_to_disk, result['photos']['photo'])

  # go to the next month
  min_upload_date = min_upload_date + relativedelta(months=1)
