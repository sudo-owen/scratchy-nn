# My neural net class
import numpy as np
import matplotlib.pyplot as plt

class NN(object):

    def __init__(self, sizes, activation="sigmoid", loss="squared_loss"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.x_dim = sizes[0]
        self.b = [np.random.randn(1, y) for y in sizes[1:]]
        self.w = [np.random.randn(x, y)/np.sqrt(x)
                  for x, y in zip(sizes[: -1], sizes[1:])]
        self.activ = activ_func[activation]["f"]
        self.d_activ = activ_func[activation]["df"]
        self.loss = loss_func[loss]["f"]
        self.d_loss = loss_func[loss]["df"]
        self.loss_vals = []
    
    # Accuracy, used only for classification tasks
    def acc(self, X, Y):
        Y_diff = Y-self.predict(X, classify=True)
        err = np.count_nonzero(Y_diff)/2
        return(1-(err/len(Y)))

    # Input is a [1 x n] vector
    def predict(self, a, classify=False):
        if (a.ndim == 1):
            a = a.reshape(1, -1)
        for i in range(self.num_layers-1):
            a = self.activ(np.dot(a, self.w[i]) + self.b[i])
        # Output predictions for classification assuming one-hot encoding 
        if classify:
            a = np.argmax(a, axis=1)
        return a
    
    # Calculate the dC/dW for a vector or matrix of X,Y
    def backprop(self, x, y):
        z = []
        activs = [x]
        for i in range(self.num_layers-1):
            x = np.dot(x, self.w[i]) + self.b[i]
            z.append(x)
            x = self.activ(x)
            activs.append(x)
        deltas = []
        b_grad = []
        w_grad = []
        for i in range(len(z)):
            if i == 0:
                # This is a [batch_size x len] vector
                delta = self.d_loss(self, z[-1], activs[-1], y)
                deltas.append(delta)
            if i != 0:
                deltas.append(np.dot(deltas[i-1], self.w[-i].T)*self.d_activ(z[-i-1]))
            w_grad.append(np.dot(activs[-2-i].T, deltas[i]))
            b_grad.append(np.sum(deltas[i], axis=0))
        return w_grad,b_grad
    
    # Given a batch of X,Y, calls backprop and updates weights,b
    def update_batch(self, batch, eta):
        delta_w, delta_b = self.backprop(batch[0], batch[1])
        step_size = eta/(len(batch[0]))
        for i in range(len(delta_w)):
            self.w[i] -= step_size*delta_w[-i-1]
            self.b[i] -= step_size*delta_b[-i-1]
    
    # Calls update_batch on the entire shuffled dataset
    def train_batch(self, X, Y, batch_size, eta):
        batch_size = len(Y)/batch_size
        combined_data = np.concatenate((X,Y), axis=1)
        np.random.shuffle(combined_data)
        y_dim = Y.shape[1]
        X_batch = np.array_split(combined_data[:, 0:-y_dim], batch_size)
        Y_batch = np.array_split(combined_data[:, -y_dim:], batch_size)
        for i in range(0, len(X_batch)):
            self.update_batch((X_batch[i],Y_batch[i]), eta)
    
    # Returns current loss on a CV set
    def cv_loss(self, X, Y):
        return self.loss(self, X, Y, sum_up=True)
    
    # Runs update_batch until set limit or when loss is < threshold
    def train(self, X, Y, batch_size, eta, max_epoch, record_loss=False, test_data=None):
        self.loss_vals.clear()
        epochs = 0
        while epochs <= max_epoch:
            epochs += 1
            self.train_batch(X,Y,batch_size,eta)
            if record_loss:
                loss = self.cv_loss(X,Y)
                self.loss_vals.append(loss)
            if test_data:
                print(str(epochs) + " : %.5f" % self.acc(test_data[0], test_data[1]))
        print("Training done!")
    
    # Prints out a graph of loss values
    def show_loss(self):
        plt.plot(self.loss_vals)
        plt.show()


##################
# Activation/Loss Functions and Utilities:
##################
# Training utilities:
def shuffle(x, y):
    new_arr = np.concatenate((x, y), axis=1)
    np.random.shuffle(new_arr)
    y_dim = y.shape[1]
    new_X = new_arr[:, :-y_dim]
    new_Y = new_arr[:, -y_dim:]
    return (new_X, new_Y)

# Neural net utilities
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return 1.7159*np.tanh(0.67*z)
def d_tanh(z):
    return (1.7159/1.5)*(1-(np.tanh(0.67*z)**2))

def arctan(z):
    return np.arctan(z)
def d_arctan(z):
    return 1/(1+z**2)

def softplus(z):
    return np.log(1+np.exp(z))
def d_softplus(z):
    return sigmoid(z)

def sin(z):
    return np.sin(z)
def cos(z):
    return np.cos(z)

def relu(z):
    return np.maximum(z, 0, z)
def d_relu(z):
    z[z<0] = 0
    z[z>1] = 1
    return z

activ_func = {
    "sigmoid": {
        "f": sigmoid,
        "df": d_sigmoid
    },
    "softplus": {
        "f": softplus,
        "df": d_softplus
    },
    "tanh": {
        "f": tanh,
        "df":d_tanh
    },
    "arctan": {
      "f": arctan,
      "df": d_arctan
    },
    "sin": {
        "f": sin,
        "df": cos
    },
    "relu": {
        "f": relu,
        "df": d_relu
    }
}

def squared_loss(nn, a, y, sum_up=False, classify=False):
    pred_y = nn.predict(a, classify)
    diff = pred_y-y
    cost = 0.5*np.linalg.norm(diff, axis=1).reshape(-1,1)**2
    if sum_up:
        cost = np.sum(cost)
    return cost
def d_squared_loss(nn, z, a, y):
    return (a-y)*nn.d_activ(z)

# Unsure if this works
# TODO: Test!
def cross_entropy(nn, a, y, sum_up=False, classify=False):
    pred_y = nn.predict(a, classify)
    cost = np.nan_to_num(-y*np.log(pred_y)-(1-y)*np.log(1-pred_y))
    if sum_up:
        cost = np.sum(cost)
    return cost
def d_cross_entropy(nn, z, a, y):
    return (a-y)

loss_func = {
    "squared_loss": {
        "f": squared_loss,
        "df": d_squared_loss
    },
    "cross_entropy": {
        "f": cross_entropy,
        "df": d_cross_entropy
    }
}