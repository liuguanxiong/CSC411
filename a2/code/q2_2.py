'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(i_digits,axis=0)
    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        difference = i_digits - means[i]
        covariances[i] = (1/difference.shape[0])*np.dot(difference.transpose(),difference)+0.01*np.identity(difference.shape[1])
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_diag = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        log_diag.append(np.log(cov_diag.reshape(8,8)))
        # ...
    all_concat = np.concatenate(log_diag, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig("2.pdf")
    plt.show()
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    likelihood = np.zeros((digits.shape[0],10))
    d = digits.shape[1]
    for i in range(10):
        sigma_det = np.linalg.det(covariances[i])
        sigma_inv = np.linalg.inv(covariances[i])
        for j in range(digits.shape[0]):
            difference = (digits[j]-means[i]).reshape(1, d)
            log_p = (-d/2)*np.log(2*np.pi)-(1/2)*np.log(sigma_det)-(1/2)*(np.matmul(np.matmul(difference,sigma_inv),difference.transpose()))
            likelihood[j,i] = log_p
    return likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative = generative_likelihood(digits, means, covariances)
    px = (1/10)*np.sum(np.exp(generative),axis=1)
    log_px = np.log(px)
    return generative + np.log(0.1) - log_px.reshape(-1, 1)


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    acc = 0
    for i in range(digits.shape[0]):
        acc += cond_likelihood[i,int(labels[i])]
    # Compute as described above and return
    return acc/digits.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def classification_accuracy(predict_labels, true_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct = 0
    for i in range(len(true_labels)):
        if predict_labels[i] == int(true_labels[i]):
            correct += 1
    return correct/len(true_labels)*100

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    plot_cov_diagonal(covariances)
    print("=================== Train data =======================")
    avg_cond_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("The average conditional log likelihood for training set is: {}".format(avg_cond_likelihood))
    print("=================== Test data =======================")
    avg_cond_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("The average conditional log likelihood for test set is: {}".format(avg_cond_likelihood))
    print("=================== Train data =======================")
    predict_labels = classify_data(train_data, means, covariances)
    print("The accuracy on the training set is {}%".format(classification_accuracy(predict_labels, train_labels)))
    print("=================== Test data =======================")
    predict_labels = classify_data(test_data, means, covariances)
    print("The accuracy on the test set is {}%".format(classification_accuracy(predict_labels, test_labels)))
if __name__ == '__main__':
    main()