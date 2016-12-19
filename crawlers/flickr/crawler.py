import datetime
import flickrapi
import requests

# TODO handle duplicates
# TODO better way to avoid pictures not being photos
# TODO what to do about the aspect ratio?
# TODO number of favorites is not a very good approximation of aesthetics of the photo

# start measuring time
start_time = datetime.datetime.now()

# settings
DOWNLOAD_LOCATION = '/home/p307k07/Code/MSc/MSc/data/'
PAGE_SIZE = 500  # Flickr allows maximum 500 photos per page
NUMBER_OF_PHOTOS_TO_GET = 100000
TAGS = ','.join(['landscape', 'portrait', 'architecture', 'macro', 'street', 'travel', 'nature', 'wildlife', 'night',
        'blackandwhite'])  # we limit the results to only these tags to avoid documents etc
NUMBER_OF_UNIQUE_RESULTS = 4000
PHOTO_SIZE = 'm'  # resolution of the downloaded photos - 'm' means small, 240 on longest side

# read API key and secret
with open('api_key.txt') as file:
    api_key = file.read()
with open('api_secret.txt') as file:
    api_secret = file.read()

# connect to Flickr
flickr = flickrapi.FlickrAPI(api_key, api_secret)

# Flickr returns only 4000 unique results, so we need to do multiple queries
# to make the results more random, we select random months and download the pictures
number_of_downloaded_photos = 0
min_upload_date = datetime.datetime(2012, 1, 1)
# while number_of_downloaded_photos < NUMBER_OF_PHOTOS_TO_GET:
while min_upload_date <= datetime.datetime.now():
    # random_year = random.randint(2010, 2016)
    # random_month = random.randint(1, 12)
    max_upload_date = min_upload_date + datetime.timedelta(days=30)
    result = flickr.photos.search(api_key=api_key, min_upload_date=min_upload_date, max_upload_date=max_upload_date,
                                  media='photos', content_type=1, tags=TAGS, per_page=PAGE_SIZE, tag_mode='any',
                                  format='parsed-json')
    number_of_new_photos = len(result['photos']['photo'])
    number_of_downloaded_photos += number_of_new_photos
    print('Downloading ' + str(number_of_new_photos) + ' photos from ' + min_upload_date.strftime('%m/%Y') + '...')
    for photo in result['photos']['photo']:
        farm_id = photo['farm']
        server_id = photo['server']
        photo_id = photo['id']
        secret = photo['secret']
        url = 'https://farm{}.staticflickr.com/{}/{}_{}_{}.jpg'.format(farm_id, server_id, photo_id, secret, PHOTO_SIZE)
        fav_result = flickr.photos.getFavorites(api_key=api_key, photo_id=photo_id, format='parsed-json')
        stars = fav_result['photo']['total']
        print('Downloading from ' + url + '... (' + stars + ' stars photo)')
        file_name = DOWNLOAD_LOCATION + stars + '_' + str(farm_id) + '_' + server_id + '_' + photo_id + '_' + secret + \
                    '.jpg'
        with open(file_name, 'wb') as file:
            response = requests.get(url)
            file.write(response.content)
    min_upload_date = min_upload_date + datetime.timedelta(days=30)

print('Done! The whole process took ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds.')
