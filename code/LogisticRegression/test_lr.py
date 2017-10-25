# -*- coding:utf8 -*-

from train import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


file_path = './data.txt'
x, y = load_data(file_path)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
print train_x.shape
print train_y.shape
print test_x.shape
print test_y.shape


def evaluate(test_x,test_y, weights):
    numSample = test_x.shape[0]
    total_acc = 0
    pred = sigmoid(np.matmul(test_x, weights))
    pred = (pred > 0.5).astype(int)
    for i in range(numSample):
         if pred[i][0] == test_y[i]: total_acc += 1
    acc = float(total_acc)/numSample

    return acc


param = {'steps': 500,
         'learning_rate': 0.01,
         'optimizer': 'smoothgd'
         }
weights = train(train_x, train_y, param)
acc = evaluate(test_x, test_y, weights)

print acc

showLogRegres(weights, test_x, test_y)

lr = LogisticRegression()
lr.fit(train_x, train_y)

pred = lr.predict(test_x)
print accuracy_score(test_y, pred)
