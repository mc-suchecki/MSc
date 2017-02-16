"""Displays a histogram for photos metadata - number of stars and views."""
import datetime
from math import log
import matplotlib.pyplot as pyplot
import numpy

# settings
PHOTOS_LIST_LOCATION = '../data/list.txt'
NUMBER_OF_BINS = 100
VIEWS_THRESHOLD = 0
DESIRED_WIDTH = 240
DESIRED_HEIGHT = 180


def is_photo_resolution_wrong(photo_metadata: list):
  width = int(photo_metadata[3])
  height = int(photo_metadata[4])
  return width != DESIRED_WIDTH or height != DESIRED_HEIGHT


# init
photos_list = open(PHOTOS_LIST_LOCATION, 'r')
stars_list = []
stars_log_list = []
views_list = []
views_log_list = []
stars_views_ratio_list = []
stars_views_log_ratio_list = []
today_timestamp = int(datetime.date.today().strftime("%s"))

# collect data about number of stars and views across the dataset
# line in the list file looks like: ID, stars, views, width, height
photos_count = 0
zero_stars_photos_count = 0
number_of_selected_photos = 0
for line in photos_list:
  photos_count += 1
  photo_metadata_list = line.split(',')
  stars = int(photo_metadata_list[1])
  views = int(photo_metadata_list[2])
  upload_date_timestamp = int(photo_metadata_list[5])
  days_since_upload = abs(today_timestamp - upload_date_timestamp)/60/60/24
  # skip photos with views below certain threshold or with inappropriate resolution
  if views < VIEWS_THRESHOLD or is_photo_resolution_wrong(photo_metadata_list):
    continue
  stars_views_ratio = 0 if (views == 0) else (stars / views)
  stars_list.append(stars)
  stars_log_list.append(log((stars + 1)/days_since_upload, 2))
  views_list.append(views)
  views_log_list.append(log((views + 1)/days_since_upload, 2))
  stars_views_ratio_list.append(stars_views_ratio)
  stars_views_log_ratio_list.append(log((stars + 1) / (views + 1), 2))
  zero_stars_photos_count += 1 if stars == 0 else 0
  number_of_selected_photos += 1

# print some basic information
print('Analyzed {} photos.'.format(photos_count))
print('{} photos left after (views > {}) and (resolution == {}x{}) selection.'.format(number_of_selected_photos,
                                                                                      VIEWS_THRESHOLD, DESIRED_WIDTH,
                                                                                      DESIRED_HEIGHT))

print('Stars stats:')
print('{}% of photos have 0 stars.'.format((zero_stars_photos_count / photos_count) * 100))
print('Average number of stars is {}.'.format(numpy.average(stars_list)))
print('Median for number of stars is {}.'.format(numpy.median(stars_list)))
print('Median for normalized number of stars is {}.'.format(numpy.median(stars_log_list)))
print('85% photos have less than {} stars.'.format(numpy.percentile(stars_list, 85)))
print('90% photos have less than {} stars.'.format(numpy.percentile(stars_list, 90)))
print('95% photos have less than {} stars.'.format(numpy.percentile(stars_list, 95)))

print('Views stats:')
print('Average number of views is {}.'.format(numpy.average(views_list)))
print('Median for number of views is {}.'.format(numpy.median(views_list)))
print('Median for normalized number of views is {}.'.format(numpy.median(views_log_list)))
print('85% photos have less than {} views.'.format(numpy.percentile(views_list, 85)))
print('90% photos have less than {} views.'.format(numpy.percentile(views_list, 90)))
print('95% photos have less than {} views.'.format(numpy.percentile(views_list, 95)))

print('Stars/Views ratio stats:')
print('Average ratio of stars to views is {}.'.format(numpy.average(stars_views_ratio_list)))
print('Median for ratio of stars to views is {}.'.format(numpy.median(stars_views_ratio_list)))
print('85% photos have stars to views ratio less than {}.'.format(numpy.percentile(stars_views_ratio_list, 85)))
print('90% photos have stars to views ratio less than {}.'.format(numpy.percentile(stars_views_ratio_list, 90)))
print('95% photos have stars to views ratio less than {}.'.format(numpy.percentile(stars_views_ratio_list, 95)))

print('Aesthetics score stats:')
print('Average is {}.'.format(numpy.average(stars_views_log_ratio_list)))
print('Median is {}.'.format(numpy.median(stars_views_log_ratio_list)))
print('85% photos have score of less than {}.'.format(numpy.percentile(stars_views_log_ratio_list, 85)))
print('90% photos have score of less than {}.'.format(numpy.percentile(stars_views_log_ratio_list, 90)))
print('95% photos have score of less than {}.'.format(numpy.percentile(stars_views_log_ratio_list, 95)))

# plot the data using histograms
fig, axes = pyplot.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
fig.suptitle("Statistics for {} photos with {} views threshold".format(number_of_selected_photos, VIEWS_THRESHOLD))
axes[0][0].set_ylabel('Number of photos')
axes[0][1].set_ylabel('Number of photos')
axes[0][2].set_ylabel('Number of photos')
axes[1][0].set_ylabel('Number of photos')
axes[1][1].set_ylabel('Number of photos')
axes[1][2].set_ylabel('Number of photos')

axes[0][0].hist(stars_list, bins=NUMBER_OF_BINS)
axes[0][0].set_xlabel('Number of stars per photo')
axes[1][0].hist(stars_log_list, bins=NUMBER_OF_BINS)
axes[1][0].set_xlabel('Logarithm of number of stars per photo')

axes[0][1].hist(views_list, bins=NUMBER_OF_BINS)
axes[0][1].set_xlabel('Number of views per photo')
axes[1][1].hist(views_log_list, bins=NUMBER_OF_BINS)
axes[1][1].set_xlabel('Logarithm of number of views per photo')

axes[0][2].hist(stars_views_ratio_list, bins=NUMBER_OF_BINS)
axes[0][2].set_xlabel('Ratio of stars to views per photo')
axes[1][2].hist(stars_views_log_ratio_list, bins=NUMBER_OF_BINS)
axes[1][2].set_xlabel('Logarithm of ratio of stars to views per photo')

pyplot.show()
