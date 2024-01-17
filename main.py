from matplotlib import pyplot as plt
import numpy as np

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

    Ix = conv(img, Kx)
    Iy = conv(img, Ky)

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

            for k in range(size):
                if (starty + k) < 0 or (starty + k) >= len(image):
                    continue
                for p in range(size):
                    if (startx + p) < 0 or (startx + p) >= len(row):
                        continue
                    total += kernal[k][p] * image[starty + k][startx + p]

            output[i][j] = total

    return output


image = plt.imread('redFlower.jpg') #read in the image
image = rgb2gray(image) # turn the image to gray scale

gaussian = gaussian_kernal(3,1) # create the gaussian kernal
image = conv(image,gaussian) #convolve the image to blur
image, theta = sobel_filters(image) #apply the sobel filter to find edges
plt.imshow(image, 'gray')

plt.show()