# -*- coding:utf8 -*-

"""
python+numpy 手动实现Logistics Regression
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    train_x = []
    train_y = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            train_x.append([float(arr[0]), float(arr[1]), 1.0])
            train_y.append(int(arr[2]))
    return np.array(train_x), np.array(train_y)


def train_test_split(x, y, test_size):
    if len(x) != len(y):
        print 'error!the size of data is not equal,please make sure!'
        return
    size = len(x)
    ## method1
    # flag = np.random.rand(size)<test_size
    # train_x = x[flag]
    # train_y = y[flag]
    # test_x = x[~flag]
    # test_y = y[~flag]

    ## method 2
    flag = np.arange(size)
    np.random.shuffle(flag)
    train_x = x[flag[:int(size*(1-test_size))]]
    train_y = y[flag[:int(size*(1-test_size))]]
    test_x = x[flag[int(size*(1-test_size)):]]
    test_y = y[flag[int(size*(1-test_size)):]]

    return train_x, test_x, train_y, test_y


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def train(train_x, train_y, param):
    if train_x.shape[0] != train_y.shape[0]:
        print 'the length of training data and label is not equal!'
    numSamples, numFeatures = train_x.shape[0], train_x.shape[1]
    weights = np.ones((numFeatures, 1))

    steps = param['steps']
    optimizer = param['optimizer']
    learning_rate = param['learning_rate']
    # 随机梯度下降
    if optimizer == 'sgd':
        for i in range(steps):
            for j in range(numSamples):
                y_output = sigmoid(np.matmul(train_x[j, :].reshape([1, -1]), weights))
                error = train_y[j]-y_output
                weights += learning_rate*train_x[j, :].reshape([-1, 1])*error
    # 普通梯度下降
    elif optimizer == 'gd':
        for i in range(steps):
            output = sigmoid(np.matmul(train_x, weights))
            train_y = train_y.reshape([-1,1])
            error = train_y - output
            weights += learning_rate * np.matmul(train_x.T, error)
    # 改进的随机梯度下降，1,学习率衰减，2,随机挑选样本
    elif optimizer == 'smoothgd':
        for i in range(steps):
            learning_rate = learning_rate*0.99
            for j in range(numSamples):
                y_output = sigmoid(np.matmul(train_x[j, :].reshape([1, -1]), weights))
                error = train_y[j]-y_output
                weights += learning_rate*train_x[j, :].reshape([-1, 1])*error

    return weights


def showLogRegres(weights, train_x, train_y):
    numSamples, numFeatures = train_x.shape[0],train_x.shape[1]
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    for i in xrange(numSamples):
        if int(train_y[i]) == 0:
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')
        elif int(train_y[i]) == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')

    min_x = min(train_x[:, 0])
    max_x = max(train_x[:, 0])
    y_min_x = float(-weights[2] - weights[0] * min_x) / weights[1]
    y_max_x = float(-weights[2] - weights[0] * max_x) / weights[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()