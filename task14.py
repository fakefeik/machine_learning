from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans


image = img_as_float(imread('parrots.jpg'))
