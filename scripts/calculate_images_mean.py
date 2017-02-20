import numpy
import os
import pyprind
import sys
from PIL import Image

SOURCE_DIR = "../data/train/"

# access all images in directory
allfiles = os.listdir(SOURCE_DIR)
images = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]

# assuming all images are the same size, get dimensions of first image
width, height = Image.open(SOURCE_DIR + images[0]).size
images_count = len(images)

# create a numpy array of floats to store the average (assume RGB images)
mean = numpy.zeros((height, width, 3), numpy.float)

# build up average pixel intensities, casting each image as an array of floats
print("Calculating mean of {} images...".format(images_count))
bar = pyprind.ProgBar(images_count, stream=sys.stdout, width=100)
for filename in images:
  image = numpy.array(Image.open(SOURCE_DIR + filename), dtype=numpy.float)
  bar.update()
  if image.ndim != 3:
    images_count -= 1
    continue
  mean += image / images_count

# round values in array and cast as 8-bit integer
mean = numpy.array(numpy.round(mean), dtype=numpy.uint8)

# generate, save and preview final image
numpy.save(SOURCE_DIR + "mean.npy", mean)
mean_image = Image.fromarray(mean, mode="RGB")
mean_image.save(SOURCE_DIR + "mean.jpg")
mean_image.show()
print("Done.")
