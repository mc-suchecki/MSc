from dateutil.relativedelta import relativedelta
import datetime
import flickrapi
import os
import requests

# TODOs
# must have
# TODO what to do about the aspect ratio?
# TODO add multi-threading to increase download speed

# maybe
# TODO think about different size
# TODO better way to avoid pictures not being photos

# settings
# DOWNLOAD_LOCATION = '/home/p307k07/Code/MSc/MSc/data/'        # laboratory
DOWNLOAD_LOCATION = '/home/mc/Code/Python/MSc/data/'  					# ThinkPad
PAGE_SIZE = 500  # Flickr allows maximum 500 photos per page
TAGS = ','.join(['landscape', 'portrait', 'architecture', 'macro', 'street', 'travel', 'nature', 'wildlife', 'night',
								 'blackandwhite'])  # we limit the results to only these tags to avoid documents etc
EXTRAS = ','.join(['views'])  # download also number of views
PHOTO_SIZE = 'm'  # resolution of the photos - 'm' means small, 240 on longest side


# functions
def get_flickr_photo_url(farm_id, server_id, photo_id, secret):
	return 'https://farm{}.staticflickr.com/{}/{}_{}_{}.jpg'.format(farm_id, server_id, photo_id, secret, PHOTO_SIZE)


def download_flickr_photo(photo_file_name, photo_url):
	with open(photo_file_name, 'wb') as photo_file:
		response = requests.get(photo_url)
		photo_file.write(response.content)


def get_photo_favorites(id):
	fav_result = flickr.photos.getFavorites(api_key=api_key, photo_id=id, format='parsed-json')
	return fav_result['photo']['total']


# THE SCRIPT

# read API key and secret
with open('api_key.txt') as file:
	api_key = file.read()
with open('api_secret.txt') as file:
	api_secret = file.read()

# connect to Flickr
flickr = flickrapi.FlickrAPI(api_key, api_secret)

# Flickr returns only 4000 unique results, so we need to do multiple queries, here we go by month
min_upload_date = datetime.datetime(2012, 1, 1)
while min_upload_date <= datetime.datetime.now():
	max_upload_date = min_upload_date + relativedelta(months=1)
	result = flickr.photos.search(api_key=api_key, min_upload_date=min_upload_date, max_upload_date=max_upload_date,
																media='photos', content_type=1, tags=TAGS, per_page=PAGE_SIZE, tag_mode='any',
																extras=EXTRAS, format='parsed-json')

	print('Downloading ' + str(len(result['photos']['photo'])) + ' photos from ' + min_upload_date.strftime('%m/%Y') + '...')
	for photo in result['photos']['photo']:
		photo_id = photo['id']
		file_name = DOWNLOAD_LOCATION + str(photo_id) + '.jpg'

		# check if photo was already downloaded
		if os.path.isfile(file_name):
			print('Skipping ' + file_name + '... (already downloaded)')
			continue

		url = get_flickr_photo_url(photo['farm'], photo['server'], photo_id, photo['secret'])

		# download number of stars and get number of views
		favorites = get_photo_favorites(photo_id)
		views = photo['views']

		# download the photo
		print('Downloading from ' + url + '... (' + favorites + ' stars, ' + views + ' views)')
		download_flickr_photo(file_name, url)

		# save the photo ID to a list along with number of stars and views
		with open(DOWNLOAD_LOCATION + 'list.txt', "a") as list_file:
			list_file.write(photo_id + ',' + favorites + ',' + views + '\n')

	# go to the next month
	min_upload_date = min_upload_date + relativedelta(months=1)
