import datetime
import flickrapi
import os
import requests

# must have
# TODO what to do about the aspect ratio?
# TODO add multi-threading to increase download speed

# maybe
# TODO think about different size
# TODO better way to avoid pictures not being photos

# start measuring time
start_time = datetime.datetime.now()

# settings
DOWNLOAD_LOCATION = '/home/p307k07/Code/MSc/MSc/data/'
PAGE_SIZE = 500                             # Flickr allows maximum 500 photos per page
TAGS = ','.join(['landscape', 'portrait', 'architecture', 'macro', 'street', 'travel', 'nature', 'wildlife', 'night',
        'blackandwhite'])                   # we limit the results to only these tags to avoid documents etc
EXTRAS = ','.join(['views'])                # download also number of views
PHOTO_SIZE = 'm'                            # resolution of the photos - 'm' means small, 240 on longest side

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
    # TODO improve this line (jump by month, not 30 days)
    max_upload_date = min_upload_date + datetime.timedelta(days=30)
    result = flickr.photos.search(api_key=api_key, min_upload_date=min_upload_date, max_upload_date=max_upload_date,
                                  media='photos', content_type=1, tags=TAGS, per_page=PAGE_SIZE, tag_mode='any',
                                  extras=EXTRAS, format='parsed-json')
    number_of_new_photos = len(result['photos']['photo'])
    print('Downloading ' + str(number_of_new_photos) + ' photos from ' + min_upload_date.strftime('%m/%Y') + '...')
    for photo in result['photos']['photo']:
        # check if photo was already downloaded
        photo_id = photo['id']
        file_name = DOWNLOAD_LOCATION + str(photo_id) + '.jpg'
        if os.path.isfile(file_name):
            print('Skipping ' + file_name + '... (already downloaded)')
            continue

        # build download URL
        farm_id = photo['farm']
        server_id = photo['server']
        secret = photo['secret']
        url = 'https://farm{}.staticflickr.com/{}/{}_{}_{}.jpg'.format(farm_id, server_id, photo_id, secret, PHOTO_SIZE)

        # download number of stars and get number of views
        fav_result = flickr.photos.getFavorites(api_key=api_key, photo_id=photo_id, format='parsed-json')
        stars = fav_result['photo']['total']
        views = photo['views']

        # download the photo
        print('Downloading from ' + url + '... (' + stars + ' stars, ' + views + ' views)')
        with open(file_name, 'wb') as photo_file:
            response = requests.get(url)
            photo_file.write(response.content)

        # save the photo ID to a list along with number of stars and views
        with open(DOWNLOAD_LOCATION + 'list.txt', "a") as list_file:
            list_file.write(photo_id + ',' + stars + ',' + views + '\n')

    # TODO improve this line (jump by month, not 30 days)
    min_upload_date = min_upload_date + datetime.timedelta(days=30)

print('Done! The whole process took ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds.')
