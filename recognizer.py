import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

sig = lambda t: 1/(1+np.exp(-t))

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


def train(X, y, epoch, path_size, layer_1_w, layer_2_w, layer_3_w):
    mse_array = np.empty((0))
    for epoch in xrange(0, epoch):
        for j in xrange(0, path_size):
            x = X[j, np.newaxis]
            layer_1 = sig(np.dot(x, layer_1_w))
            layer_2 = sig(np.dot(layer_1, layer_2_w))
            layer_3 = sig(np.dot(layer_2, layer_3_w))



            layer_3_delta = (layer_3 - y[j, np.newaxis])*(layer_3)*(1-layer_3)
            layer_2_delta = np.dot(layer_3_delta, layer_3_w.T) * (layer_2)*(1-layer_2)
            layer_1_delta = np.dot(layer_2_delta, layer_2_w.T) * (layer_1)*(1-layer_1)

            layer_3_w -= np.dot(layer_2.T, layer_3_delta)
            layer_2_w -= np.dot(layer_1.T, layer_2_delta)
            layer_1_w -= np.dot(x.T, layer_1_delta)
        mse = ((y[j, np.newaxis] - layer_3) ** 2).mean(axis=1)
        mse_array = np.append(mse_array, mse)
    return mse_array



def test(test_x, layer_1_w, layer_2_w, layer_3_w):
    test_layer_1 = sig(np.dot(test_x, layer_1_w))
    test_layer_2 = sig(np.dot(test_layer_1, layer_2_w))
    test_layer_3 = sig(np.dot(test_layer_2, layer_3_w))
    print(test_layer_3)



def runMLPClasssifier(X, y, test_image, max_iter):
    mlp = MLPClassifier(hidden_layer_sizes=(7, 6), activation='logistic')
    mlp.max_iter = max_iter
    mlp.fit(X, y)

    predictions = mlp.predict_proba([test_image])
    print(predictions)

def runBackProp(X, y, epoch, train_paths, layer_1_w, layer_2_w, layer_3_w, test_image):
    mse_array = train(X, y, epoch, train_paths, layer_1_w, layer_2_w, layer_3_w)
    test(test_image, layer_1_w, layer_2_w, layer_3_w)
    #showPlot(mse_array)

def showPlot(y_axis):
    x_axis = np.arange(1, 2001, 1);
    plt.plot(x_axis, y_axis, 'ro')
    plt.show()

def refreshWeights(layer_1_w, layer_2_w, layer_3_w):
    layer_1_w = np.random.rand(901,7)
    layer_2_w = np.random.rand(7,6)
    layer_3_w = np.random.rand(6,4)

############### MAIN ##############

X = np.zeros((20, 901)) # Array full of input
y = np.array([ [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
               [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0],
               [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
               [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1] ])

# Train paths
train_paths = np.array(['d_1.png', 'd_2.png', 'd_3.png', 'd_4.png', 'd_5.png',
                        'e_1.png', 'e_2.png', 'e_3.png', 'e_4.png', 'e_5.png',
                        'm_1.png', 'm_2.png', 'm_3.png', 'm_4.png', 'm_5.png',
                        'i_1.png', 'i_2.png', 'i_3.png', 'i_4.png', 'i_5.png'])

# Test paths
test_paths = np.array(['d_6.png', 'd_7.png', 'd_8.png', 'd_9.png', 'd_10.png',
                       'e_6.png', 'e_7.png', 'e_8.png', 'e_9.png', 'e_10.png',
                       'm_6.png', 'm_7.png', 'm_8.png', 'm_9.png', 'm_10.png',
                       'i_6.png', 'i_7.png', 'i_8.png', 'i_9.png', 'i_10.png'])

# Test labels
test_labels = np.array([ [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
                         [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0],
                         [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                         [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1] ])

for i in xrange(0, train_paths.size):
    a = image2Vector('images/train/' + train_paths[i])
    X[i] = a


layer_1_w = np.zeros((901,7))
layer_2_w = np.zeros((7,6))
layer_3_w = np.zeros((6,4))


for i in xrange(0, test_paths.size):
    refreshWeights(layer_1_w, layer_2_w, layer_3_w)
    test_image = image2Vector('images/test/' + test_paths[i])
    test_label = test_labels[i]
    runBackProp(X, y, 2000, train_paths.size, layer_1_w, layer_2_w, layer_3_w, test_image)


#runBackProp(X, y, 2000, paths.size, layer_1_w, layer_2_w, layer_3_w, test_image)
#runMLPClasssifier(X, y, test_image, 2000)
