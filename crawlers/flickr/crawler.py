import flickrapi

# read API key and secret
with open('api_key.txt') as file:
    api_key = file.read()
with open('api_secret.txt') as file:
    api_secret = file.read()

# connect to Flickr
flickr = flickrapi.FlickrAPI(api_key, api_secret)

# test
photos = flickr.photos.search(api_key=api_key)
print(photos)
