import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    n = X.shape[0]
    XtXW = np.dot(np.dot(np.transpose(X), X), w)
    Xty = np.dot(np.transpose(X), y)
    return (2*XtXW - 2*Xty)/n

def squared_distance(vec1, vec2):
    '''
    Compute the squared euclidean distance between two vectors
    '''
    return np.sum((vec2-vec1)**2)

def mean_mini_grad(w, K, batch_sampler):
    '''
    Compute mini-batch mean
    '''
    acc = 0
    for k in range(K):
        X_b, y_b = batch_sampler.get_batch()
        acc += lin_reg_gradient(X_b, y_b, w)
    return acc/K

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    K = 500
    m = 400
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    
    mean_gradient = mean_mini_grad(w, K, batch_sampler)
    lin_gradient = lin_reg_gradient(X, y, w)
    
    print("squared distance: ", squared_distance(mean_gradient, lin_gradient))
    print("cosine similarity: ", cosine_similarity(mean_gradient, lin_gradient))
    
    sd = np.arange(400)
    for i in range(1,m+1):
        gradient = np.arange(K)
        for k in range(K):
            X_b, y_b = batch_sampler.get_batch(i)
            gradient[k] = lin_reg_gradient(X_b, y_b, w)[0]
        sd[i-1] = np.std(gradient,ddof = 1)
       
    m = np.arange(400)
    m = m + 1
    plt.plot(log(m),log(sd))
    ax = plt.gca()
    ax.grid()
    plt.xlabel("log(m)")
    plt.ylabel("log(stdev)")
    plt.savefig("q3.pdf")
    
    
    
if __name__ == '__main__':
    main()
