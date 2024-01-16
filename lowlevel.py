from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve

def rgb2gray(image):

    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale_image


def gaussian_kernal(size, sigma = 1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    gaussian = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return gaussian

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def conv(image, kernal):
    output = image.copy()
    middle = len(kernal)//2
    size = len(kernal)
    for i, row in enumerate(image):
        for j, ele in enumerate(row):
            startx = j - middle
            starty = i - middle
            total = 0
            divide = 0
            for k in range(size):
                if (starty + k) < 0 or (starty + k) >= len(image):
                    continue
                for p in range(size):
                    if (startx + p) < 0 or (startx + p) >= len(row):
                        continue
                    total += kernal[k][p] * image[starty + k][startx + p]
                    divide += kernal[k][p]
            output[i][j] = total / divide

    return output

image = plt.imread('sudo1.png')
print(len(image))
print(image[0][0][0])

print(image.ndim)
output = image.copy()
image = rgb2gray(image)
print(image)
gaussian = gaussian_kernal(3,1)
#print(gaussian)
image = conv(image,gaussian)
image, theta = sobel_filters(image)
plt.imshow(image, 'gray')

plt.show()