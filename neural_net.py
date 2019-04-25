
# coding: utf-8

# Three-layer Neural Network classifier

# In[69]:


import numpy as np
import matplotlib.pyplot as plt


# In[70]:


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


# In[71]:


class ThreeLayerNet(object):
  """
  A three-layer fully-connected neural network. The net has an input dimension of
  N, two-hidden layers of dimension H, and performs classification on one class.
  We train the network with a square loss function.  The network uses the sigmoid
  activation function at each hidden layer.  

  In other words, the network has the following architecture:

  input -> fully connected layer -> sigmoid -> fully connected layer -> sigmoid -> 
    fully connected layer -> square loss

  The output of the third fully-connected layer is the prediction.
  """

  def __init__(self, input_size, hidden_size, std=1.0):
    """
    Initialize the model. Weights are initialized to zero and
    biases are initialized to one. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, H)
    b2: Second layer biases; has shape (H,)
    W3: Third layer weights; has shape (H,)
    b3: Third layer bias; scalar

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, 1)
    self.params['b3'] = 0

  def loss(self, X, y=None):
    """
    Compute the loss and gradients for a three layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the set {-1, 1} This parameter is optional; if it
      is not passed then we only return the prediction, and if it is passed then we
      instead return the loss and gradients.

    Returns:
    If y is None, return a vector of scores of shape (N) where scores[i] is
    the score for input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss for this batch of training samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    H1 = np.dot(X,W1) + b1
    SH1 = sigmoid(H1)
    H2 = np.dot(SH1, W2) + b2
    SH2 = sigmoid(H2)
    scores = np.dot(SH2, W3) + b3

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    loss = np.sum(0.5 * (scores - y)**2)
    
    # Backward pass: compute gradients
    grads = {}
    y = y.reshape((N,1))
    
    dOut = scores - y
    dOut /= N
                    
    # Gradients for add gate adding H2*W3 + b3
    db3 = np.sum(dOut, axis=0)
    grads['b3'] = db3 
                        
    # Gradient for multiply gate multiplying H2 and W3           
    dW3 = SH2.T.dot(dOut) 
    grads['W3'] = dW3 
    dOut = dOut.reshape((N,1))
    W3 = W3.reshape(hidden_size, 1)
    dH2 = np.dot(dOut,W3.T)
                         
    # Gradient for Sigmoid activation function at H2
    dS2 = SH2 * (1 - SH2) * dH2
    
    # Gradients for add gate adding H1*W2 + b2
    db2 = np.sum(dS2, axis=0)
    grads['b2'] = db2 
                          
    # Gradient for multiply gate multiplying H1 and W2           
    dW2 = SH1.T.dot(dS2) 
    grads['W2'] = dW2
    dH1 = W2.T.dot(dS2.T) 
    
    # Gradient for Sigmoid activation function at H1
    dS1 = SH1 * (1-SH1) * dH1.T
                         
    # Gradient for add gate adding X*W1 + b1
    db1 = np.sum(dS1, axis=0)
    grads['b1'] = db1  
         
    # Gradient for multiply gate multiplying X and W1           
    dW1 = X.T.dot(dS1) 
    grads['W1'] = dW1 
   
    return loss, grads

  def train(self, X, y, gamma=0.05, d=1., num_epochs=100, verbose=False, batch_size=16):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels.
    - gamma: Scalar representing a hyperparameter in the learning rate schedule
    - d: Scalar representing a hyperparameter in the learning rate schedule
    - num_epochs: Number of iterations to take when optimizing.
    - verbose: boolean; if true print progress during optimization.
    - batch_size: Int representing the batch_size for SGD
    """
    N, D = X.shape

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    t = 1.0
    
    learning_rate = gamma 
    for epoch in range(num_epochs):
      
    #Randomly shuffle data
        index = np.random.permutation(N)
        X = X[index]
        y = y[index]
        learning_rate = gamma / (1 + gamma / d * t)
        t += 1
        
        #Iterate through randomly shuffled training examples
        start = 0
        end = batch_size
        
        #Iterate through randomly shuffled training examples
        while(end < N):
            x_batch = X[start:end]
            y_batch = y[start:end].reshape(-1,1)
            
            # Compute loss and gradients using the current item
            loss, grads = self.loss(x_batch, y_batch)
            loss_history.append(loss)
            self.params['W1'] += -learning_rate * grads['W1'] 
            self.params['b1'] += -learning_rate * grads['b1'] 
            self.params['W2'] += -learning_rate * grads['W2'] 
            self.params['b2'] += -learning_rate * grads['b2'] 
            self.params['W3'] += -learning_rate * grads['W3'] 
            self.params['b3'] += -learning_rate * grads['b3'] 
            
            start += batch_size
            end += batch_size
            
        if verbose and epoch % 10 == 0:
            print('iteration %d / %d: loss %f' % (epoch, num_epochs, loss))
        
    return loss_history
    

  def predict(self, X):
    """
    Use the trained weights of this three-layer network to predict labels for
    data points. 
    
    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. 
    """
    y_pred = None
    y = self.loss(X)
    y_pred = np.sign(y) 
    return y_pred




# In[72]:


def load_data(path, add_bias=False):
    """
    Loads and processes the bank note data set
    
    Inputs:
    -path:  string representing the path of the file
    -add_bias:  boolean representing whether an extra column of ones is added to the data, representing
                a bias value
    
    Returns:
    -X:  a numpy array of shape [no_samples, no_attributes (+1 if add_bias is True)]
    -y:  a numpy array of shape [no_samples] that represents the labels {-1, 1} for the dataset X
    """
    
    import numpy as np
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            example = line.strip().split(',')
            if len(example) > 0:
                example = [float(i) for i in example]
                data.append(example)
    X = np.array(data, dtype=np.float64)
    y = X[:,-1]
    y = y.astype(int)
    y[y == 0] = -1
    X = X[:,:-1]
    
    if add_bias == True:
        bias = np.ones((X.shape[0],1),dtype=np.float64) 
        X = np.hstack((X,bias))

    return X, y


# In[73]:


X_train, y_train = load_data('/Users/janaanlake/Documents/CS_5350/HW3/bank-note/train.csv', add_bias=True)
X_test, y_test = load_data("/Users/janaanlake/Documents/CS_5350/HW3/bank-note/test.csv", add_bias=True)

input_size = X_train.shape[1]

print("Results for initializing weights from Guassian distribution:")
print('\n')
for hidden_size in [5,10,25,50,100]:
    print("Results using hidden layer size of " + str(hidden_size) + " :")
    net = ThreeLayerNet(input_size, hidden_size)
    # Train the network
    loss_history = net.train(X_train, y_train, num_epochs=100, verbose=False)
    
    # Predict on the training set
    y_train = y_train.reshape((-1,1))
    train_acc = (net.predict(X_train) == y_train).mean()
    print("The average training error is " + "{0:.2f}".format((1.0-train_acc)*100) + "%")
    
    #Predict on the testing set
    y_test = y_test.reshape((-1,1))
    test_acc = (net.predict(X_test) == y_test).mean()
    print("The average testing error is " + "{0:.2f}".format((1.0-test_acc)*100) + "%")

    #Plot the loss function 
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    plt.show()


# In[74]:


#Run the same tests setting weight parameters to zero
print("Results for weight parameters initialized at zero:")
print('\n')
for hidden_size in [5,10,25,50,100]:
    print("Results using hidden layer size of " + str(hidden_size) + " :")
    net = ThreeLayerNet(input_size, hidden_size, std=0)
    # Train the network
    loss_history = net.train(X_train, y_train, num_epochs=5, verbose=False)
    
    # Predict on the training set
    y_train = y_train.reshape((y_train.shape[0],1))
    train_acc = (net.predict(X_train) == y_train).mean()
    print("The average training error is " + "{0:.2f}".format((1.0-train_acc)*100) + "%")
    
    #Predict on the testing set
    y_test = y_test.reshape((y_test.shape[0],1))
    test_acc = (net.predict(X_test) == y_test).mean()
    print("The average testing error is " + "{0:.2f}".format((1.0-test_acc)*100) + "%")

    #Plot the loss function 
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    plt.show()


# In[53]:


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 2
hidden_size = 2
num_inputs = 5

def init_toy_model():
    return ThreeLayerNet(input_size, hidden_size, std=1e-1)

def init_toy_data():
    X = np.array([[1,1]])
    y = np.array([1])
    return X, y

net = init_toy_model()
net.params['W1'] = np.array([[-2,2],[-3,3]])
net.params['b1'] = np.array([-1,1])
net.params['W2'] = np.array([[-2,2],[-3,3]])
net.params['b2'] = np.array([-1,1])
net.params['W3'] = np.array([2,-1.5])
net.params['b3'] = -1
X, y = init_toy_data()


# In[22]:


scores = net.loss(X)
print('Your score:')
print(scores)
print()
print('correct scores:')
print(-2.437)
print()


# In[23]:


loss, _ = net.loss(X, y)
correct_loss = 5.906

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

