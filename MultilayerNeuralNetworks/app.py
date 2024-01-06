import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from time import perf_counter
import pickle
import logging
logging.basicConfig(level=logging.INFO)

# plot the activation function (sigmoid)
z = np.linspace(-10, 10, 100)
sigma = 1/(1+np.exp(-z))
fig = plt.figure(figsize=(6,4))
ax = fig.subplots()
ax.plot(z, sigma, linestyle='-', linewidth=0.5, color='k')
ax.axhline(0.5, linestyle='--', linewidth=0.5, color='k')
ax.axvline(0., linestyle='--', linewidth=0.5, color='k')
ax.set_xlabel('z')
ax.set_ylabel(r'$\sigma\left(z\right)$')
loc_arrow = (0.4, 1/(1+np.exp(-0.4)))
ax.annotate(r'$\sigma\left(z\right)=\frac{1}{1+e^{-z}}$', loc_arrow, xytext=(-5.,0.9),
            fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='orange', ec='black', lw=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=0.5, shrinkB=0.5))
fig.savefig('images/sigmoid_activation.png', dpi=600)

def sigmoid(z):
    '''
    Sigmoid (logistic) activation function
    :param z: input value
    :return: sigmoid(input value)
    '''
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    '''
    One-hot-encoding of a list of classes.
    :param y: list with classes
    :param num_labels: number of classes
    :return: one-hot-encoding of the input list of classes
    '''
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

class MultilayerNeuralNetClassifier:
    '''
    A generalised, multilayer, feedforward, fully connected neural network
    '''

    def __init__(self, layers: list[int], seed: int=1):
        '''
        Initialises the weights and bias terms of the neural network layers
        :param layers: list of integers with the number of nodes in each layer. The first integer
        corresponds to the number of features and the last to the number of classes to be predicted. For example,
        layers = [784, 50, 10] means the input vector has 784 elements, the first (and only) hidden layers has 50
        nodes and the output is a vector of length 10 (number of classes)
        :param seed:
        '''
        super().__init__()

        self.n_classes = layers[-1]
        self.n_features = layers[0]
        self.n_layers = layers

        # initialise the random number generator
        rng = np.random.RandomState(seed)

        # set the weights and bias terms of all layers
        self.W = []
        self.b = []
        for ilayer in range(len(self.n_layers)-1):
            W = rng.normal(loc=0.0,
                           scale=0.1,
                           size=(self.n_layers[ilayer+1], self.n_layers[ilayer])
                           )
            b = np.zeros(shape=self.n_layers[ilayer+1])
            logging.info(f'set up layer {ilayer} with {self.n_layers[ilayer]}x{self.n_layers[ilayer+1]} weights and {self.n_layers[ilayer+1]} bias terms')
            self.W.append(W)
            self.b.append(b)

    def forward(self, x):
        '''
        Forward pass for a single sample or a batch of samples
        :param x: input with shape (N, n^0)
        :return: list with the matrices with the activated values of each layer. The last element of the list
        are the predicted values with shape (N, n^L)
        '''
        a_previous = x
        a = []
        for ilayer in range(len(self.n_layers) - 1):
            z = np.dot(a_previous, self.W[ilayer].T) + self.b[ilayer]
            a_previous = sigmoid(z)
            a.append(a_previous)
        return a

    def loss_accuracy(self, x, y):
        '''
        Computes the loss function and the accuracy for a single sample or batch of samples
        :param x: input with shape (N, n^0)
        :param y: known classes with shape is (N,)
        :return: loss
        '''
        output = self.forward(x)[-1]
        loss = (output - int_to_onehot(y, self.n_classes))**2
        loss = loss.mean()

        predicted_classes = np.argmax(output, axis=1)
        accuracy = np.mean(predicted_classes==y)
        return loss, accuracy

    def backward(self, x, a, y):
        '''
        Backward pass for a single sample or a batch of samples
        :param x: input with shape (N, n^0)
        :param a: list with activated values from each layer
        :param y: known classes with shape (N,)
        :return: tuple with two lists, the first list contains the derivatives of the loss function
                 with regard to the weight matrices and the second list contains the derivatives of the loss function
                 with regard to the bias terms
        '''
        dW = []
        db = []
        for ilayer in range(len(self.n_layers)-2, -1, -1):
            # weights of L-1 layer (last layer)
            if ilayer == len(self.n_layers)-2:
                d_loss_d_AL = 2*(a[ilayer]-int_to_onehot(y, self.n_classes))/(x.shape[0]*self.n_classes)
                d_loss_d_ZL = d_loss_d_AL*a[ilayer]*(1.-a[ilayer])
                d_loss_dWL_1 = np.dot(d_loss_d_ZL.T, a[ilayer-1])
                d_loss_dbL_1 = d_loss_d_ZL.sum(0)
                dW.append(d_loss_dWL_1)
                db.append(d_loss_dbL_1)
            # weights of intermediate layers
            elif ilayer > 0:
                d_loss_d_AL = np.dot(d_loss_d_ZL, self.W[ilayer+1])
                d_loss_d_ZL = d_loss_d_AL*a[ilayer]*(1.-a[ilayer])
                d_loss_dWL_k_1 = np.dot(d_loss_d_ZL.T, a[ilayer-1])
                d_loss_dbL_k_1 = d_loss_d_ZL.sum(0)
                dW.append(d_loss_dWL_k_1)
                db.append(d_loss_dbL_k_1)
            # weights of 0 layer (first layer)
            else:
                d_loss_d_AL = np.dot(d_loss_d_ZL, self.W[ilayer+1])
                d_loss_d_ZL = d_loss_d_AL*a[ilayer]*(1.-a[ilayer])
                d_loss_dWL_k_1 = np.dot(d_loss_d_ZL.T, x)
                d_loss_dbL_k_1 = d_loss_d_ZL.sum(0)
                dW.append(d_loss_dWL_k_1)
                db.append(d_loss_dbL_k_1)
        return dW[::-1], db[::-1] # put the lists in forward order


    def backward_numerical(self, x, y, k):
        '''
        Backward pass for a single sample or a batch of samples. The gradients are computed numerically with central
        differences. This function is only to be used for testing the correctness of the analytically computed
        gradients.
        :param x: input with shape (N, n^0)
        :param y: known classes with shape (N,)
        :param k: layer to compute the gradients for (k in [0, n^L-2])
        :return: tuple with two lists, the first list contains the derivatives of the loss function
                 with regard to the weight matrices and the second list contains the derivatives of the loss function
        '''

        # perturb the weights of layer k
        epsilon = 1.e-5
        dW = np.zeros_like(self.W[k])
        db = np.zeros_like(self.b[k])
        for i1 in range(dW.shape[0]):
            # numerical gradients of weights
            for i2 in range(dW.shape[1]):
                w_orig = self.W[k][i1, i2]
                self.W[k][i1, i2] = w_orig + epsilon
                loss_f, _ = self.loss_accuracy(x, y)
                self.W[k][i1, i2] = w_orig - epsilon
                loss_b, _ = self.loss_accuracy(x, y)
                dW[i1, i2] = (loss_f - loss_b)/(2*epsilon)
                self.W[k][i1, i2] = w_orig
            # numerical gradients of bias terms
            b_orig = self.b[k][i1]
            self.b[k][i1] = b_orig + epsilon
            loss_f, _ = self.loss_accuracy(x, y)
            self.b[k][i1] = b_orig - epsilon
            loss_b, _ = self.loss_accuracy(x, y)
            db[i1] = (loss_f - loss_b)/(2*epsilon)
            self.b[k][i1] = b_orig
        return dW, db

# check the correctness of the gradients
model = MultilayerNeuralNetClassifier(layers=[784, 50, 40, 30, 10])
N = 7
x = np.random.random(size=(N, 784))
y = np.random.randint(low=0, high=2, size=N)
loss, _ = model.loss_accuracy(x, y)
a = model.forward(x)
dW, db = model.backward(x, a, y)
# check the weights and bias terms of the last layer
for k in [3, 2, 1, 0]:
    dWn, dbn = model.backward_numerical(x, y, k=k)
    print(f'layer {k}: {np.isclose(dW[k],dWn).sum()} out of {dWn.size} weight gradients are numerically equal')
    print(f'layer {k}:{np.isclose(db[k], dbn).sum()} out of {dbn.size} bias term gradients are numerically equal')

# obtain the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto', as_frame=False)
print(f'original X: {X.shape=}, {X.dtype=}, {X.min()=}, {X.max()=}')
print(f'original y: {y.shape=}, {y.dtype=}')
# convert X to floats in the range [-1, 1]
X = 2.*(X/255.) - 1.
# convert y to integers
y = y.astype(int)
# check shapes, datatypes and distribution of classes
print(f'processed X: {X.shape=}, {X.dtype=}, {X.min()=}, {X.max()=}')
print(f'processed y: {y.shape=}, {y.dtype=}')
print(f'class counts: '+', '.join([f'{row[0]}:{row[1]}' for row in np.array(np.unique(y, return_counts=True)).T]))
# .. pickle the dataset in case it is not available in the future
with open('datasets/mnist_784.pickle', 'wb') as f:
    pickle.dump((X, y), f)


# visualise ten images per digit to see the variation in hand writing
# .. shuffle the images to select samples randomly
idxs = np.arange(len(X))
np.random.shuffle(idxs)
fig = plt.figure(figsize=(10, 10))
fig, axs = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
idxs = np.arange(len(X))
for iax, ax in enumerate(axs.ravel()):
    digit = iax%10
    isample = iax//10
    logging.info(f'plotting digit {digit}, sample {isample}')
    img = X[y[idxs]==digit][isample].reshape(28, 28)
    ax.imshow(img, cmap='Greys')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('off')
    ax.set_title(f'true label: {y[y[idxs]==digit][isample]}', fontsize=4)
fig.tight_layout()
fig.savefig('images/random_samples.png', dpi=600)

# split into training set, external (hold-out) test set and validation set
seed = 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=10_000)



# batch generator
def generate_batches(X, y, rng, batch_size=100):
    '''
    Generates batches of given size
    :param X: feature matrix
    :param y: class vector
    :param rng: NumPy random number generator for reproducibility
    :param batch_size: number of samples in the batch
    :return: tuple with features and class vector
    '''
    idxs = np.arange(X.shape[0])
    rng.shuffle(idxs)
    for idx_start in range(0, X.shape[0], batch_size):
        idxs_batch = idxs[idx_start:  idx_start+batch_size]
        if len(idxs_batch) == batch_size:
            yield X[idxs_batch], y[idxs_batch]

# check that the batch generator returns batches of correct size
seed = 100
rng = np.random.RandomState(100)
for X_batch, y_batch in generate_batches(X_train, y_train, rng, batch_size=501):
    print(f'{X_batch.shape=}, {y_batch.shape=}')


# starting accuracy
seed = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=10_000)
model = MultilayerNeuralNetClassifier(layers=[784, 50, 10], seed=seed)
loss, accuracy = model.loss_accuracy(X_test, y_test)
print(f'starting accuracy {accuracy: 0.3f}')


# train the model
def train(model, X_train, y_train, X_test, y_test, num_epochs=10, learning_rate=0.1, batch_size=100, seed=100):
    '''
    Trains a model and reports the loss and accuracy for both the training set
    and the external (hold-out) test set
    :param model: neural network model
    :param X_train: feature matrix (training set)
    :param y_train: label vector (training set)
    :param X_test: feature matrix (external, hold-out test set)
    :param y_test: class vector (external, hold-out test set)
    :param num_epochs: number passes through the training set (epochs)
    :param learning_rate: learning rate for updating the weights
    :param batch_size: batch size
    :return: list of tuples with the loss and accuracy for the training set and the loss and accuracy for the external
    (hold-out) test set
    '''
    convergence = []
    # initiate the random number generator for the batch generator
    rng = np.random.RandomState(seed)
    for epoch in range(num_epochs):
        batch_generator = generate_batches(X_train, y_train, batch_size=batch_size, rng=rng)
        loss_training = 0
        accuracy_training = 0
        for ibatch, (X_batch, y_batch) in enumerate(batch_generator):

            # forward pass
            a = model.forward(X_batch)

            # backward pass
            dW, db = model.backward(X_batch, a, y_batch)

            # update the model parameters (weights and bias terms)
            for k in range(len(dW)):
                model.W[k] += -learning_rate*dW[k]
                model.b[k] += -learning_rate * db[k]

            loss_batch, accuracy_batch = model.loss_accuracy(X_batch, y_batch)
            loss_training += loss_batch
            accuracy_training += accuracy_batch
        # log the epoch
        loss_training = loss_training/ibatch
        accuracy_training = accuracy_training /ibatch
        loss_test, accuracy_test = model.loss_accuracy(X_test, y_test)
        convergence.append((loss_training, accuracy_training, loss_test, accuracy_test))
        print(f'epoch {epoch}: {loss_training=:0.3f} | {accuracy_training=:0.3f} | {loss_test=:0.3f} | {accuracy_test=:0.3f}')
    return convergence

# train the model
seed = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=10_000)
model = MultilayerNeuralNetClassifier(layers=[784, 50, 10], seed=seed)
loss, accuracy = model.loss_accuracy(X_test, y_test)
convergence = train(model, X_train, y_train, X_test, y_test, num_epochs=100, learning_rate=0.1, batch_size=200, seed=seed)



# plot training set loss and its rate of change
x = np.arange(len(convergence))
losses_training = np.array(convergence)[:,0]
diff_losses_training = np.gradient(losses_training)
fig = plt.figure(figsize=(6, 4))
ax = fig.subplots()
ax2 = ax.twinx()
# .. plot the training set loss on the left axis
ax.plot(x, losses_training,
        color='k', linestyle='-', marker='s', markersize=4,
        markerfacecolor='none', linewidth=0.5, markeredgewidth=0.5,
        label='training loss'
        )
ax.set_ylabel('training loss', color='k', fontsize=8)
ax.set_xlabel('epoch')
# .. plot the gradient of the training set loss on the right axis
ax2.set_yscale('log')
ax2.plot(x, -diff_losses_training,
         color='k', linestyle='-', marker='o', markersize=4,
         markerfacecolor='none', linewidth=0.5, markeredgewidth=0.5,
         label='log(training loss reduction per epoch)')
ax2.set_ylabel('log(training loss reduction per epoch)', fontsize=8)
ax.legend(loc='upper left', fontsize=5, frameon=False)
ax2.legend(loc='upper right', fontsize=5, frameon=False)
ax.tick_params(axis='both', labelsize=8)
ax2.tick_params(axis='both', labelsize=8)
fig.tight_layout()
fig.savefig('images/training_loss_convergence.png', dpi=600)


# plot training and external (hold-out) test set accuracy
x = np.arange(len(convergence))
accuracy_training = np.array(convergence)[:, 1]
accuracy_test = np.array(convergence)[:, 3]
fig = plt.figure(figsize=(6, 4))
ax = fig.subplots()
# .. plot the training set accuracy
ax.plot(x, accuracy_training,
        color='k', linestyle='--', linewidth=0.5,
        label='training set accuracy'
        )
ax.set_ylabel('accuracy', color='k', fontsize=8)
ax.set_xlabel('epoch')
# .. plot the external (hold-out) test set accuracy
ax.plot(x, accuracy_test,
        color='k', linestyle='-', linewidth=0.5,
        label='external (hold-out) test set accuracy'
        )
ax.legend(loc='upper left', fontsize=5, frameon=False)
ax.tick_params(axis='both', labelsize=8)
fig.tight_layout()
fig.savefig('images/training_test_accuracy.png', dpi=600)


# cross-validation and hyperparameter tuning
seed = 100
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=10_000)
num_folds = 6
cross_validation = []
for n_hidden_layers in [1, 2, 3]:
    for n_hidden_nodes in range(10, 60, 10):
        for learning_rate in [0.1, 0.2, 0.3]:
            for i_fold in range(0, 6):
                start = i_fold * X_temp.shape[0] // num_folds
                end = (i_fold + 1) * X_temp.shape[0] // num_folds if i_fold < num_folds - 1 else X_temp.shape[0]
                idxs_validation = np.zeros_like(y_temp)
                idxs_validation[start:end] = 1
                idxs_validation = idxs_validation.astype(bool)
                X_validation = X_temp[idxs_validation]
                y_validation = y_temp[idxs_validation]
                X_train = X_temp[~idxs_validation]
                y_train = y_temp[~idxs_validation]
                seed = 100
                model = MultilayerNeuralNetClassifier(layers=[784]+[n_hidden_nodes]*n_hidden_layers+[10], seed=seed)
                start_time = perf_counter()
                convergence = train(model, X_train, y_train, X_validation, y_validation, num_epochs=250, learning_rate=learning_rate, batch_size=200, seed=seed)
                convergence = np.array(convergence)
                res = {'n_hidden_layers': n_hidden_layers,
                       'n_hidden_nodes': n_hidden_nodes,
                       'learning_rate': learning_rate,
                       'fold': i_fold,
                       'training loss': convergence[-1, 0],
                       'training loss gradient (mean 10 last epochs)': np.gradient(convergence[:, 0])[-10].mean(),
                       'training accuracy': convergence[-1, 1],
                       'validation loss': convergence[-1, 2],
                       'validation accuracy': convergence[-1, 3],
                       'training time': perf_counter() - start_time
                       }
                print(res)
                cross_validation.append(res)
cross_validation = pd.DataFrame(cross_validation)
cross_validation.to_excel(f'output/cross_validation.xlsx')

# find the optimal model parameters
optimal_network_parameters = cross_validation.groupby(['n_hidden_layers', 'n_hidden_nodes', 'learning_rate'])['validation accuracy'].mean().idxmax()
best_mean_cv_accuracy = cross_validation.groupby(['n_hidden_layers', 'n_hidden_nodes', 'learning_rate'])['validation accuracy'].mean().max()
print(f'optimal parameters: n_hidden_layers={optimal_network_parameters[0]}, '
      f'n_hidden_nodes={optimal_network_parameters[1]}, '
      f'learning rate={optimal_network_parameters[2]:0.1f}')
print(f'best mean cross validation accuracy: {best_mean_cv_accuracy:0.3f}')

# effect of number of hidden layers and number of nodes, using largest learning rate
hypersurface = (cross_validation
                .loc[cross_validation['learning_rate']==0.3]
                .groupby(['n_hidden_layers', 'n_hidden_nodes'])['validation accuracy'].mean()
                .unstack(level=1))
print(hypersurface.to_markdown())

# fit the model using the whole X_temp, y_temp
seed = 100
model = MultilayerNeuralNetClassifier(layers=[784, 50, 10], seed=seed)
convergence = train(model, X_temp, y_temp, X_test, y_test, num_epochs=1000, learning_rate=0.3, batch_size=200, seed=seed)


