"""Displays a histogram for photos metadata - number of stars and views."""
import matplotlib.pyplot as pyplot

# settings
PHOTOS_LIST_LOCATION = '../data/test/list.txt'
NUMBER_OF_BINS = 100

# init
photos_list = open(PHOTOS_LIST_LOCATION, 'r')
stars_list = []
views_list = []
stars_views_ratio_list = []

# collect data about number of stars and views across the dataset
# line in the list file looks like: ID, stars, views, width, height
for line in photos_list:
  photo_metadata_array = line.split(',')
  stars = int(photo_metadata_array[1])
  views = int(photo_metadata_array[2])
  stars_views_ratio = 0 if (views == 0) else (stars / views)
  stars_list.append(stars)
  views_list.append(views)
  stars_views_ratio_list.append(stars_views_ratio)

# plot the data using histograms
fig, axes = pyplot.subplots(nrows=1, ncols=3, sharex=False, sharey=True)
axes[0].set_ylabel('Number of photos')
axes[1].set_ylabel('Number of photos')
axes[2].set_ylabel('Number of photos')
axes[0].hist(stars_list, bins=NUMBER_OF_BINS)
axes[0].set_xlabel('Number of stars per photo')
axes[1].hist(views_list, bins=NUMBER_OF_BINS)
axes[1].set_xlabel('Number of views per photo')
axes[2].hist(views_list, bins=NUMBER_OF_BINS)
axes[2].set_xlabel('Ratio of stars to views per photo')
pyplot.show()
