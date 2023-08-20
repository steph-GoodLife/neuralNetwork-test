import numpy as np

#https://www.youtube.com/watch?v=cWGbRQGTWRE

def sigmoid(x):
    return 1/(1+np.exp(-x))

def neural_network(inputs, target, epochs, lr):
    w1 = np.random.rand(2,4)
    w2 = np.random.rand(4,1)

    for epoch in range(epochs):
        layer1 = sigmoid(np.dot(inputs, w1))
        output = sigmoid(np.dot(layer1, w2))

        error = target - output
        delta2 = 2 * error * output * (1 - output)
        delta1 = delta2.dot(w2.T) * layer1 * (1 - layer1)

        w2 += lr * layer1.T.dot(delta2)
        w1 += lr * inputs.T.dot(delta1)

    return np.round(output.squeeze())

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

print("OR", neural_network(inputs, np.array([[0], [1], [1], [1]]), 5000, 0.1))


