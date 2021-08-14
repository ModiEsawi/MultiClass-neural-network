import numpy as np
import sys
import random


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# softmax activation function
def softmax(A):
    A = A - np.max(A)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


#  normalize the training sets using the min-max normalization
def normalizeSet(trainingSet, maxVals, minVals):
    newMin = 0
    newMax = 1
    for trainingSample in trainingSet:
        for column in range(trainingSet.shape[1]):
            if maxVals[column] != minVals[column]:  # prevent dividing by zero
                if trainingSample[column] > maxVals[column]:
                    trainingSample[column] = 1
                    continue
                elif trainingSample[column] < minVals[column]:
                    trainingSample[column] = 0
                    continue
                normalizedValue = (((trainingSample[column] - minVals[column]) / (maxVals[column] - minVals[column]))
                                   * (newMax - newMin)) + newMin
            else:
                normalizedValue = 1
            trainingSample[column] = normalizedValue


# set the weights, biases and learning rate of the neural network
def neuralNetwork():
    attributes = 784
    hidden_nodes = 100
    output_labels = 10
    global w1
    w1 = np.random.uniform(-0.5, 0.5, (attributes, hidden_nodes))
    global b1
    b1 = np.random.uniform(-1, 1, hidden_nodes)
    global w2
    w2 = np.random.uniform(-0.5, 0.5, (hidden_nodes, output_labels))
    global b2
    b2 = np.random.uniform(-1, 1, output_labels)
    global learningRate
    learningRate = 0.01


# train the module using forward and back propagation
def train(givenSample):
    global w1
    global b1
    global w2
    global b2
    # forward propagation
    z1 = np.dot(givenSample, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    # back propagation
    dL_dz2 = a2 - one_hot_labels
    dz2_dw2 = a1
    dL_w2 = np.dot(dz2_dw2.T, dL_dz2)
    dL_b2 = dL_dz2
    dz2_da1 = w2
    dL_da1 = np.dot(dL_dz2, dz2_da1.T)
    da1_dz1 = sigmoid_der(z1)
    dz1_dw1 = givenSample
    dL_w1 = np.dot(dz1_dw1.T, da1_dz1 * dL_da1)
    dL_b1 = dL_da1 * da1_dz1
    # update the weights
    global learningRate
    w1 -= learningRate * dL_w1
    b1 -= learningRate * dL_b1.sum(axis=0)
    w2 -= learningRate * dL_w2
    b2 -= learningRate * dL_b2.sum(axis=0)
    return a2


# read the train_x train_y files
train_x = np.loadtxt(sys.argv[1])
train_y = np.loadtxt(sys.argv[2])
maxValues = np.max(train_x, axis=0)  # max values in each column
minValues = np.min(train_x, axis=0)  # min values in each column
normalizeSet(train_x, maxValues, minValues)
neuralNetwork()
one_hot_labels = None
output = None
# train the module for 28 epochs
for epoch in range(28):
    zipped = list(zip(train_x, train_y))
    random.shuffle(zipped)
    train_x, train_y = zip(*zipped)
    for sample in range(len(train_x)):
        feature_set = np.vstack([train_x[sample]])
        one_hot_labels = np.zeros((1, 10))
        one_hot_labels[0, int(train_y[sample])] = 1
        output = train(feature_set)
# test the module on a given test set and write the module predictions to a file
test_x = np.loadtxt(sys.argv[3])
normalizeSet(test_x, maxValues, minValues)
outputFile = open("test_y", "w")
for testSample in range(len(test_x)):
    global w1
    global b1
    global w2
    global b2
    feature_set = np.vstack([test_x[testSample]])
    zh = np.dot(feature_set, w1) + b1
    ah = sigmoid(zh)
    zo = np.dot(ah, w2) + b2
    ao = softmax(zo)
    answer = np.argmax(ao, axis=1)[0]
    outputFile.write(str(answer) + "\n")
outputFile.close()
