import cv2
import numpy as np

def image2Vector(path):
    # Load image and turn into grayscaled image
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert image to binary image
    ret, binary_image = cv2.threshold(gray_image, 127, 1, cv2.THRESH_BINARY)
    image_array = np.array(binary_image)

    # reshape array and append 1
    image_array = np.ravel(image_array)
    image_array = np.append(image_array, 1)
    return image_array


X = np.zeros((20, 901)) # Array full of input
# Train paths
paths = np.array(['d_1.png', 'd_2.png', 'd_3.png', 'd_4.png', 'd_5.png',
                  'e_1.png', 'e_2.png', 'e_3.png', 'e_4.png', 'e_5.png',
                  'm_1.png', 'm_2.png', 'm_3.png', 'm_4.png', 'm_5.png',
                  'i_1.png', 'i_2.png', 'i_3.png', 'i_4.png', 'i_5.png'])
for i in xrange(0, paths.size):
    a = image2Vector('images/train/' + paths[i])
    X[i] = a
print X
