#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:51:44 2017

@author: Guanxiong
"""

'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

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
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        # loop over all digits and get k nearest neighbour
        dist = self.l2_distance(test_point)
        neighbours = []
        while k > 0:
            if k > 1:
                smallest = dist.argmin()
                dist[smallest] = float('inf')
            else:
                smallest = self.tie_broken(test_point, dist)

            neighbours.append(int(self.train_labels[smallest]))
            k -= 1
        digit = np.argmax(np.bincount(np.array(neighbours)))
        return digit

    def tie_broken(self,test_point,dist):
        '''
        Break the tie based on the absolute difference of the two points

        return the index of the point that has minimum absolute difference
        with the test_point
        '''

        # check whether there is tie
        optimal = dist.argmin()
        optimal_val = dist[optimal]
        dist[optimal] = float('inf')

        # deal with tie
        while dist.min() == optimal_val:
            candidate = dist.argmin()
            dist[candidate] = float('inf')
            if self.train_labels[optimal] == self.train_labels[candidate]:
                continue
            optimal_img = self.train_data[optimal]
            candidate_img = self.train_data[candidate]
            difference1 = sum(np.abs(optimal_img - test_point))
            difference2 = sum(np.abs(candidate_img - test_point))
            if difference2 < difference1:
                print("pruned")
                optimal = candidate
        return optimal

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    optimal = 0
    optimal_val = 0
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...

        # randomly partition the data into 10 folders
        idx = [i for i in range(0,len(train_data))]
        np.random.shuffle(idx)
        interval = round(len(train_data)/10)
        accuracy = 0
        for i in range(0, 10):
            valid_idx = idx[interval*i: interval*(i+1)]
            train_idx = list(set(idx) - set(valid_idx))
            cur_train_data = train_data[train_idx]
            cur_train_label = train_labels[train_idx]
            cur_valid_data = train_data[valid_idx]
            cur_valid_label = train_labels[valid_idx]
            knn = KNearestNeighbor(cur_train_data, cur_train_label)
            accuracy += classification_accuracy(knn, k, cur_valid_data, cur_valid_label)
        accuracy /= 10
        if accuracy > optimal_val:
            optimal = k
            optimal_val = accuracy
    return optimal,optimal_val


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct = 0
    for index in range(len(eval_data)):
        predicted_label = knn.query_knn(eval_data[index],k)
        correct += predicted_label == eval_labels[index]
    return correct/len(eval_data)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    print("=================== Test on single data =======================")
    predicted_label = knn.query_knn(test_data[0], 50)
    print("The labeled value={} || The predicted label={}".format(test_labels[0],predicted_label))
    print("\n")

    print("=================== Test on Test data =======================")
    accuracy = classification_accuracy(knn,100,test_data, test_labels)
    print("The accuracy on the Test Data = {}".format(accuracy))
    print("\n")

    print("=================== Test on Optimal K =======================")
    optimal_k = cross_validation(train_data,train_labels)
    print("The optimal K={}".format(optimal_k))

if __name__ == '__main__':
    main()