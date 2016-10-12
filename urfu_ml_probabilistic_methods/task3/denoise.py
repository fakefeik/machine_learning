import numpy as np
from copy import deepcopy
from skimage.io import imread
from matplotlib.image import imsave
from matplotlib.pyplot import show, imshow


def sub(v1, v2):
    return v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]


def mult(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def is_in_field(src, x, y):
    return 0 <= x < len(src) and 0 <= y < len(src[0])


def get_neighbours(src, x, y, eight_neighbours=True):
    coords = (
        ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        if eight_neighbours else ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    for i, j in coords:
        if is_in_field(src, x + i, y + j):
            yield src[x + i][y + j]


def get_neighbour_colors(src, x, y, eps=2):
    s = set()
    neighbours = [src[x][y]] + list(get_neighbours(src, x, y))
    for neighbour in neighbours:
        for i in range(-eps, eps + 1):
            for j in range(-eps, eps + 1):
                for k in range(-eps, eps + 1):
                    t = (neighbour[0] + i, neighbour[1] + j, neighbour[2] + k)
                    if 0 <= t[0] <= 255 and 0 <= t[1] <= 255 and 0 <= t[2] <= 255:
                        s.add(t)
    return s


def restore_image(src, covar=100, max_diff=200, weight_diff=0.02, iterations=10):
    buffer = [[], []]
    for x in range(src.shape[0]):
        buffer[0].append([])
        buffer[1].append([])
        for y in range(src.shape[1]):
            buffer[0][x].append(src[x][y])
            buffer[1][x].append(0)
    s = 1
    d = 0
    V_max = src.shape[0] * src.shape[1] * ((256 ** 2) / (2 * covar) + 4 * weight_diff * max_diff)
    max_it = iterations * src.shape[0]
    it = 0
    for i in range(iterations):
        s, d = d, s
        for r in range(src.shape[0]):
            print('\r{:.3f}%'.format(it/max_it*100), end='')
            it += 1
            for c in range(src.shape[1]):
                V_local = V_max
                min_val = -1
                for val in get_neighbour_colors(src, r, c):
                    v = sub(val, src[r][c])
                    V_data = (mult(v, v)) / (2 * covar)
                    V_diff = 0

                    for e in get_neighbours(buffer[s], r, c):
                        v = sub(val, e)
                        V_diff += min(mult(v, v), max_diff)

                    V_current = V_data + weight_diff * V_diff

                    if V_current < V_local:
                        min_val = val
                        V_local = V_current

                buffer[d][r][c] = min_val
    return buffer[d]

r = np.array(restore_image(imread('taj-rgb-noise.jpg'), iterations=20), np.uint8)
imshow(r)
show()
imsave('denoised.jpg', r)
