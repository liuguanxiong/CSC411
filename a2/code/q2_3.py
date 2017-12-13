'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.stats import beta

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        total = i_digits.shape[0]
        p = (np.sum(i_digits, axis=0) + 1)/(total+2)
        eta[i] = p
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    image=[]
    for i in range(10):
        img_i = class_images[i]
        image.append(img_i.reshape(8,8))
        # ...
    all_concat = np.concatenate(image, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig("3.pdf")
    plt.show()
def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            generated_data[i,j] = np.random.choice(2,1,p=[1-eta[i,j],eta[i,j]])

    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    likelihood = np.zeros((bin_digits.shape[0],10))
    for i in range(10):
        for j in range(bin_digits.shape[0]):
            log_p = np.sum(bin_digits[j]*np.log(eta[i]) + (1-bin_digits[j])*np.log(1-eta[i]))
            likelihood[j,i] = log_p
    return likelihood

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative = generative_likelihood(bin_digits, eta)
    px = (1/10)*np.sum(np.exp(generative),axis=1)
    log_px = np.log(px)
    return generative + np.log(0.1) - log_px.reshape(-1, 1)
def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    acc = 0
    for i in range(bin_digits.shape[0]):
        acc += cond_likelihood[i,int(labels[i])]
    # Compute as described above and return
    return acc/bin_digits.shape[0]

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    
    print("=================== Train data =======================")
    avg_cond_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    print("The average conditional log likelihood for training set is: {}".format(avg_cond_likelihood))
    print("=================== Test data =======================")
    avg_cond_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print("The average conditional log likelihood for test set is: {}".format(avg_cond_likelihood))
    print("=================== Train data =======================")
    predict_labels = classify_data(train_data, eta)
    print("The accuracy on the training set is {}%".format(classification_accuracy(predict_labels, train_labels)))
    print("=================== Test data =======================")
    predict_labels = classify_data(test_data, eta)
    print("The accuracy on the test set is {}%".format(classification_accuracy(predict_labels, test_labels)))

if __name__ == '__main__':
    main()
