import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = self.beta * self.vel + self.lr * grad
        params = params - self.vel

        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count+1)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)

        first = np.dot(self.w,np.transpose(X))
        for i in range(len(first)):
            first[i] = max(0,1-first[i]*y[i])
        return first

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)
        w = self.w[:-1]
        w = np.append(w,0)
        
        gradient = []
        first = np.dot(self.w,np.transpose(X))
        for i in range(len(first)):
            if first[i]*y[i] < 1:
                gradient.append(-y[i]*X[i])
            else:
                gradient.append(np.zeros(len(X[i])))
        gradient = np.array(gradient)
        gradient = np.mean(gradient, axis=0)
        gradient = gradient * self.c + w
        return gradient


    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)

        return np.transpose(np.where(np.dot(X,self.w)>=0, 1, -1))

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for i in range(steps):
        w_history.append(optimizer.update_params(w_history[i], func_grad(w_history[i])))
        # Optimize and update the history
    return w_history

def verify_optimizer():
    w00 = optimize_test_function(GDOptimizer(lr=1.0,beta=0.0), w_init=10.0, steps=200)
    w09 = optimize_test_function(GDOptimizer(lr=1.0,beta=0.9), w_init=10.0, steps=200)
    
    plt.plot(w00,label="beta=0.0")
    plt.plot(w09,label="beta=0.9")
    plt.legend()
    plt.show()

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty,train_data.shape[1])
    sampler = BatchSampler(train_data, train_targets, batchsize)
    for i in range(iters):
        x_batch, y_batch = sampler.get_batch(batchsize)
        grad = svm.grad(x_batch, y_batch)
        svm.w = optimizer.update_params(svm.w, grad)
    return svm


    


if __name__ == '__main__':
    
    verify_optimizer()
    
    train_data, train_targets, test_data, test_targets = load_data()
    
    optimizer00 = GDOptimizer(lr=0.05, beta=0.0)
    optimizer01 = GDOptimizer(lr=0.05, beta=0.1)   
    svm01 = optimize_svm(train_data, train_targets, 1.0, optimizer01, 100, 500)
    svm00 = optimize_svm(train_data, train_targets, 1.0, optimizer00, 100, 500)
    
    print('======================beta=0.1===========================')
    loss = (1/2)*np.dot(svm01.w,svm01.w)+svm01.c*np.mean(svm01.hinge_loss(train_data,train_targets))
    print('train loss = {}'.format(loss))
    avg_hinge = np.mean(svm01.hinge_loss(train_data,train_targets))
    print('average train hinge loss = {}'.format(avg_hinge))
    loss = (1/2)*np.dot(svm01.w,svm01.w)+svm01.c*np.mean(svm01.hinge_loss(test_data,test_targets))
    print('test loss = {}'.format(loss))
    avg_hinge = np.mean(svm01.hinge_loss(test_data,test_targets))
    print('average test hinge loss = {}'.format(avg_hinge))
    pred_train = svm01.classify(train_data)
    print('train accuracy = {}'.format((pred_train == train_targets).mean()))
    pred_test = svm01.classify(test_data)
    print('test accuracy = {}'.format((pred_test == test_targets).mean()))
    w01 = svm01.w[:-1].reshape(28, 28)
    plt.imshow(w01, cmap='gray')
    plt.show()
    
    print('======================beta=0.0===========================')
    loss = (1/2)*np.dot(svm00.w,svm00.w)+svm00.c*np.mean(svm00.hinge_loss(train_data,train_targets))
    print('train loss = {}'.format(loss))
    avg_hinge = np.mean(svm00.hinge_loss(train_data,train_targets))
    print('average train hinge loss = {}'.format(avg_hinge))
    loss = (1/2)*np.dot(svm00.w,svm00.w)+svm00.c*np.mean(svm00.hinge_loss(test_data,test_targets))
    print('test loss = {}'.format(loss))
    avg_hinge = np.mean(svm00.hinge_loss(test_data,test_targets))
    print('average test hinge loss = {}'.format(avg_hinge))
    pred_train = svm00.classify(train_data)
    print('train accuracy = {}'.format((pred_train == train_targets).mean()))
    pred_test = svm00.classify(test_data)
    print('test accuracy = {}'.format((pred_test == test_targets).mean()))
    w00 = svm00.w[:-1].reshape(28, 28)
    plt.imshow(w00, cmap='gray')
    plt.show()