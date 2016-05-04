import numpy as np
import matplotlib.image
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import compare_psnr
from sklearn.cluster import KMeans


def np_to_arr(arr):
    return [arr[0], arr[1], arr[2]]


def to_rgb(arr):
    return [int(arr[0] * 255), int(arr[1] * 255), int(arr[2] * 255)]


image = img_as_float(imread('parrots.jpg'))

l = []
for i in range(len(image)):
    for j in range(len(image[0])):
        l.append(image[i][j])
clf = KMeans(n_clusters=10, random_state=241)
clf.fit(l)
pred = clf.predict(l)

l = list(map(lambda x: x, l))
l1 = list(map(lambda x: clf.cluster_centers_[pred[x]], range(len(l))))
l2 = list(map(lambda x: to_rgb(clf.cluster_centers_[pred[x]]), range(len(l))))

print(compare_psnr(np.array(l), np.array(l1)))

count = 0
l3 = [None] * len(image)
for i in range(len(image)):
    l3[i] = [None] * len(image[i])
    for j in range(len(image[i])):
        l3[i][j] = l2[count]
        for k in range(3):
            l3[i][j][k] = 255 - l3[i][j][k]
        count += 1

l3 = np.array(l3, np.int32)
matplotlib.image.imsave('name.png', l3)
