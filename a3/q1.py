'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model, test_pred

def SVM(bow_train, train_labels, bow_test, test_labels):

    svc = svm.SVC()
    parameters = {'kernel':['linear', 'rbf'], 'C':[1, 10]}
    model = GridSearchCV(svc, parameters, cv=10)
    model.fit(bow_train, train_labels)
    print(model.best_params_)
    
    #evaluate the baseline model
    train_pred = model.predict(bow_train)
    print('SVM baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('SVM baseline test accuracy = {}'.format((test_pred == test_labels).mean()))
    
    return model, test_pred


def logistic_regression(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    model = LogisticRegression()
    model.fit(bow_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(bow_train)
    print('Logistic Regression baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('Logistic Regression baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model, test_pred

def neural_network(bow_train, train_labels, bow_test, test_labels):

    model = MLPClassifier(hidden_layer_sizes=25)
    model.fit(bow_train, train_labels)
    
    #evaluate the baseline model
    train_pred = model.predict(bow_train)
    print('Neural Network baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('Neural Network baseline test accuracy = {}'.format((test_pred == test_labels).mean()))
    
    return model, test_pred
    

def confusion_matrix(test_pred, test_label):
    
    matrix = np.zeros((20,20))
    for i in range(len(test_pred)):
        matrix[test_pred[i]][test_label[i]] += 1
    return matrix

def find_most_confused(matrix):
    
    max_count = 0
    max_i = -1
    max_j = -1
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                if matrix[i,j] > max_count:
                    max_count = matrix[i,j]
                    max_i = i
                    max_j = j
    return max_i, max_j

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test, feature_names = tf_idf_features(train_data, test_data)
    
#    bnb_model = bnb_baseline(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
#    SVM_model, SVM_pred = SVM(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
#    logistic_model, logistic_pred = logistic_regression(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    Neural_model, Neural_pred = neural_network(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    
    confusion_matrix = confusion_matrix(Neural_pred, test_data.target)
    i, j = find_most_confused(confusion_matrix)
    print(i,j)
    w = confusion_matrix.reshape(20, 20)
    plt.imshow(w, cmap='gray')
    plt.show()