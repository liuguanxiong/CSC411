from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features

def summarize(X, y, features):
    print('-----------------------------------------------------------------')
    print("Dimensions :{}".format(X.shape[1]))
    print("Features: {}".format(features))
    print("Number of Data points: {}".format(X.shape[0]))
    print("Target Min: {}".format(min(y)))
    print("Target mean: {}".format(np.mean(y)))
    print("Target standard deviation: {}".format(np.std(y)))
    print("Target Max: {}".format(max(y)))

def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.scatter(X[:,i], y)
        plt.xlabel(features[i])
        plt.ylabel('Target')
        #TODO: Plot feature i against y
    
    plt.tight_layout()
    plt.show()

def split_data(X, y, percent_training):
    training_size = int(round(X.shape[0] * percent_training))
    indices = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X_training, X_test = X[indices[:training_size],:], X[indices[training_size:],:]
    y_training, y_test = y[indices[:training_size]], y[indices[training_size:]]
    return X_training, X_test, y_training, y_test
    
def fit_regression(X,Y):
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias), axis=1)

    XtX = np.dot(np.transpose(X), X)
    XtY = np.dot(np.transpose(X), Y)
    return np.linalg.solve(XtX,XtY) #solve for W
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    #raise NotImplementedError()

def show_result(w, X_test, y_test, features):
    print('-----------------------------------------------------------------')
    print('<Feature, Weight>')
    for i in range(len(features)):
        print('<{} , {}>'.format(features[i], w[i]))
        
    print('-----------------------------------------------------------------')
    
    bias = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((X_test, bias), axis=1)
    
    y_hat = np.dot(X_test, w)
    n = X_test.shape[0]
    MSE = sum((y_hat - y_test)**2)/n
    MAE = sum(abs(y_hat - y_test))/n
    MAPE = (100/n)*sum(abs((y_test - y_hat)/y_test))
    print('MSE = {}'.format(MSE))#mean squared error
    print('MAE = {}'.format(MAE))#mean absolute error
    print('MAPE = {}'.format(MAPE))#mean absolute percentage error
    
    
    
def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    summarize(X, y, features)
    # Visualize the features
    visualize(X, y, features)
    
    #TODO: Split data into train and test
    X_training, X_test, y_training, y_test = split_data(X, y, 0.8)
    
    # Fit regression model
    w = fit_regression(X_training, y_training)
    
    
    show_result(w, X_test, y_test, features)
    
    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()

