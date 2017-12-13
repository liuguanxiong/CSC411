'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        dist = self.l2_distance(test_point)
        neighbors = np.zeros(k,dtype='int64')
        for i in range(k):
            smallest_index = dist.argmin()
            dist[smallest_index] = math.inf
            neighbors[i] = (int(self.train_labels[smallest_index])) #store the nearest
                                                                    #label first
        
        #count the frequency of each label
        counter = Counter(list(neighbors))
        most_common_fre = counter.most_common()[0][1]
        dictionary = dict(counter)
        
        #put all ties in candidate and iterate over neighbors, the first element that
        #exists in candidate is the nearest label and I choose the label as the final
        #digit(answer)
        candidate=[]
        for key in dictionary:
            if dictionary[key] == most_common_fre:
                candidate.append(key)

        for item in neighbors:
            if item in candidate:
                digit = item
                break
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    
    X = train_data
    y = train_labels
    kf = KFold(n_splits=10)
    optimal_acc = 0
    optimal_k = 0
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        total_acc = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            new_knn = KNearestNeighbor(X_train, y_train)
            total_acc += classification_accuracy(new_knn, k, X_test, y_test)
        mean_acc = total_acc/10
        if mean_acc > optimal_acc:
            optimal_acc = mean_acc
            optimal_k = k
    return optimal_k,optimal_acc

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct = 0
    for i in range(len(eval_data)):
        predicted_label = knn.query_knn(eval_data[i],k)
        if predicted_label == eval_labels[i]:
            correct += 1
    return correct/len(eval_data)*100

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)


    # Example usage:
    print("=================== Test Train data K=1 =======================")
    accuracy = classification_accuracy(knn,1,train_data, train_labels)
    print("The accuracy with K=1 for Train Data = {}%".format(accuracy))

    print("=================== Test Test data K=1 =======================")
    accuracy = classification_accuracy(knn,1,test_data, test_labels)
    print("The accuracy with K=1 for Test Data = {}%".format(accuracy))

    print("=================== Test Train data K=15 =======================")
    accuracy = classification_accuracy(knn,15,train_data, train_labels)
    print("The accuracy with K=15 for Train Data = {}%".format(accuracy))

    print("=================== Test Test data K=15 =======================")
    accuracy = classification_accuracy(knn,15,test_data, test_labels)
    print("The accuracy with K=15 for Test Data = {}%".format(accuracy))

    print("=================== Test Optimal K =======================")
    optimal_k,optimal_acc= cross_validation(train_data,train_labels)
    train_accuracy = classification_accuracy(knn,optimal_k,train_data, train_labels)
    test_accuracy = classification_accuracy(knn,optimal_k,test_data, test_labels)
    print("The optimal K={}".format(optimal_k))
    print("The average accuracy with optimal_k = {}%".format(optimal_acc))
    print("The accuracy with optimal_k for Train Data = {}%".format(train_accuracy))
    print("The accuracy with optimal_k for Test Data = {}%".format(test_accuracy))
if __name__ == '__main__':
    main()