import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def initialize_parameters(layer_units):
    # layer_units is an array containing the number of units(neurons) in each layer. ex: layer_units[0] gives the number of units in layer 0(input layer)
    
    #np.random.seed(3)
    parameters = {}
    L = len(layer_units)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_units[l], layer_units[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_units[l],1))
        
        #assert(parameters['W' + str(l)].shape == (layer_units[l], layer_units[l-1]))
        #assert(parameters['b' + str(l)].shape == (layer_units[l], 1))
        
    return parameters

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))  
    return A


def sigmoid_deravative(Z):
    s = 1/(1+np.exp(-Z))
    #assert (dZ.shape == Z.shape)
    return s


def forward_propagation(A_prev, W, b):
    # A_prev = activation from the previous layer
    # W = weights of the current layer
    # b = bias of the current layer

    # Z = W * A_prev + b
    # A = activation of the current layer
    # cache = ((A_prev, W, b), (Z))

    Z = np.dot(W,A_prev) + b
    linear_cache = (A_prev, W, b)
    A = sigmoid(Z)
    
    #assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, Z)

    return A, cache

def forward_propagation_layers(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L+1):
        A_prev = A 
        A, cache = forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)
    
    AL = A  #activation of the final layer(output)
    
    #assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
    # AL = activation of the final layer
    # cost = cross entropy cost
    
    m = Y.shape[1]

    cost = -(1/m) * np.sum((Y * np.log(AL)) + ((1-Y) * np. log(1-AL))) 
    
    # cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    # cost = np.mean(np.square(Y - np.round(AL)))
    
    #cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def back_propagation(dA, cache):
    # dA = derivative of the cost w.r.t current layer activation
    # cache = ((A_prev, W, b), Z)

    # dZ = derivative of the cost w.r.t z(linear function)
    # dA_prev = derivative of the cost w.r.t previous layer activation

    # dW = derivative of the cost w.r.t current layer weights
    # db = derivative of the cost w.r.t current layer bias

    linear_cache, Z = cache

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
        
    dZ = dA * sigmoid_deravative(Z)
    dW = (1/m) * np.dot(dZ,A_prev.T)

    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    #dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def back_propagation_layers(AL, Y, caches):
    # AL = activation of the final layer
    # Y = output labels

    derivatives = {} # dictionary containing the computed derivatives for all weights and biases
    L = len(caches) # the number of layers
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # derivative of cost w.r.t activation of the final layer
    dAL = - (Y/AL - (1 - Y)/ (1 - AL))
    # dAL = np.divide(AL - Y, (1 - AL) *  AL)
    dA = dAL
    
    # Loop from l=L-1 to l=1
    for l in reversed(range(1,L+1)):
        current_cache = caches[l-1]
        dA_prev, derivatives["dW" + str(l)], derivatives["db" + str(l)] = back_propagation(dA, current_cache)
        dA = dA_prev

    return derivatives


def update_parameters(parameters, derivatives, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * derivatives["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * derivatives["db" + str(l+1)]

    return parameters


def model(X, Y, layer_units, learning_rate = 0.005, num_iterations = 5000):
   
    #np.random.seed(1)
    costs = []
    
    # initialize weights and biases for each layer
    parameters = initialize_parameters(layer_units)
    
    for i in range(0, num_iterations):

        # forward propagation - get the activation of the final layer(output) and the additional data from each layer needed for backprop
        AL, caches = forward_propagation_layers(X, parameters)
        
        # compute the cost
        cost = compute_cost(AL, Y)
    
        # back propagation - returns the derivatives of the cost w.r.t all the parameters
        derivatives = back_propagation_layers(AL, Y, caches)

        #difference = gradient_check_n(parameters, derivatives, X, Y)
 
        # update the parameters
        parameters = update_parameters(parameters, derivatives, learning_rate)
                
        # Print the cost every 100 training example
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def predict(X, y, parameters):
    
    m = X.shape[1] # number of training examples
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # forward propagation
    output, caches = forward_propagation_layers(X, parameters)

    # convert output to 0/1 using 0.5 as threshold
    p = (output >= 0.5).astype(int)
    # print(p,y)
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


#read the data set
#data = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')
X,Y = make_moons(n_samples=2000, shuffle=True, noise=None, random_state=None)

m = X.shape[0]
Y = Y.reshape(m,1)

#remove rows with no value and remove 1st column(id)
# data = data[~np.isnan(data).any(axis=1)]
# data = np.delete(data, 0 , axis=1)

# #length of the dataset
# m = len(data)

# #seperate features from labels
# features, labels = data[:, :9], data[:, 9:]

# #rename labels so that they take binary values
# labels[labels == 2] = 0
# labels[labels == 4] = 1

# #normalize features
# X_max, X_min = features.max(), features.min()
# attributes = (features - X_min) / (X_max - X_min)

# #shuffle and divde dataset into training set and test set
# np.random.seed(3)
# np.random.shuffle(data)

print(Y.shape)
margin = m//10*8
print(margin)
X_train, X_test = X[:margin, :].T, X[margin:, :].T
Y_train, Y_test = Y[:margin, :].T, Y[margin:, :].T

#print(X, Y)

layer_units = [X_train.shape[0], 5, 3, 1]
parameters = model(X_train, Y_train, layer_units, num_iterations = 10000)
pred_train = predict(X_train, Y_train, parameters)
pred_test = predict(X_test, Y_test, parameters)
