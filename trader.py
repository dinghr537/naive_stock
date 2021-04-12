#!/usr/bin/env python
# coding: utf-8


data_train = ""
data_test = ""
data_out = ""


if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # global data_train, data_test, data_out
    data_train = args.training
    data_test = args.testing
    data_out = args.output


# In[1]:

# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[2]:


# read data
title = ['Open', 'High', 'Low', 'Close']
train_set = pd.read_csv(data_train, header=None, names = title)


# In[3]:

# data processing
def generate_data(df,lookahead_days=1):
    raw_x= []
    title=['Open', 'High', 'Low', 'Close']
    raw_x = df.loc[:,title].values
    X, Y = [], []
    n = len(df)
    for i in range(0, n-lookahead_days):
        _Y = np.mean(raw_x[i+1:i+1+lookahead_days, 0])
        Y.append(_Y)
    X = generate_feature(raw_x, 0, n-lookahead_days)
    Y = np.array(Y)
    return X, Y


# In[4]:


# feature includes current data and max,min,avg of open price of last 5 days
def generate_feature(raw_x,start,end):
    X = []
    for i in range(start, end):
        current_price = raw_x[i][0]
        avg5_price = max5_price = min5_price = 0
        if i == 0 :
            avg5_price = max5_price = min5_price = current_price     
        elif i < 5:
            max5_price = np.max(raw_x[0:i,0])
            min5_price = np.min(raw_x[0:i,0]) 
            avg5_price = np.mean(raw_x[0:i,0])         
        else:
            max5_price = np.max(raw_x[i-5:i,0])
            min5_price = np.min(raw_x[i-5:i,0])
            avg5_price = np.mean(raw_x[i-5:i,0])     
        _X = np.append(raw_x[i], [avg5_price, max5_price, min5_price])
        X.append(_X)
    return np.array(X)


# In[5]:

# generate two models, one for tomorrow, one for next 4 days
lookahead_days = 4
one_day_X, one_day_Y = generate_data(train_set,1)
lookahead_X, lookahead_Y = generate_data(train_set,lookahead_days)


# In[6]:


# use PolynomialFeatures to try more features
degree = 2
poly = PolynomialFeatures(degree)
poly_one_day_X = poly.fit_transform(one_day_X)
poly_lookahead_X = poly.fit_transform(lookahead_X)


# In[7]:


# use ridge regression
# predict_tomorrow = linear_model.Ridge(alpha=1)
# predict_3days = linear_model.Ridge(alpha=1)

# I've also tried Lasso regression but it didn't work well
# predict_tomorrow = linear_model.Lasso(alpha=1, tol=0.01)
# predict_3days = linear_model.Lasso(alpha=1, tol=0.01)

# sklearn support vector regression
# from sklearn import svm
# predict_tomorrow = svm.SVR()
# predict_3days = svm.SVR()

# finally I choose to use ransom forest with 20 decision trees each
from sklearn import ensemble
predict_tomorrow = ensemble.RandomForestRegressor(n_estimators=20)
predict_3days = ensemble.RandomForestRegressor(n_estimators=20)


# In[8]:

# fit model
predict_tomorrow.fit(poly_one_day_X ,one_day_Y)
predict_3days.fit(poly_lookahead_X ,lookahead_Y)


# In[9]:

# read in testing data
def get_testing_data(path):
    title = ['Open', 'High', 'Low', 'Close']
    df = pd.read_csv(path, header=None, names=title)
    raw_x = df.loc[:,title].values
    X = generate_feature(raw_x, 0, len(raw_x))
    return X


# In[10]:

# call get_testing_data
# just read data to memory, won't see the content of data until line 203
testing_data = get_testing_data(data_test)


# In[11]:

# use class Trader to predict prices and make decisions
class Trader():
    def __init__(self, predict_tomorrow, predict_3days):
        self.predict_tomorrow = predict_tomorrow
        self.predict_3days = predict_3days
        self.day_count = 0    
        self.slot = 0

    def predict_action(self,current_data):
        current_price = current_data[0]
        next_price = self.predict_tomorrow.predict(current_data.reshape(1,-1))
        mean_3days_price = self.predict_3days.predict(current_data.reshape(1,-1))
        self.day_count += 1
        action = self.policy(current_price, next_price, mean_3days_price)
        return action

    def policy(self, current_price, next_price, price_of_days):
        # you already have one
        if self.slot == 1: 
            if next_price > price_of_days: 
                self.slot = 0
                return '-1'
        # you have nothing
        elif self.slot == 0:
            if next_price > price_of_days: 
                self.slot = -1
                return '-1'
            if  next_price < price_of_days: 
                self.slot = 1
                return '1'
        # you owe one
        elif self.slot == -1:
            if next_price < price_of_days: 
                self.slot =0
                return '1'
        return '0'


# In[12]:


trader = Trader(predict_tomorrow, predict_3days)


# In[13]:

# generate output file
with open(data_out, 'w') as output_file:
    for line in testing_data[0:-1]:
        # read one line of testing data and output the action
        # then read the next line
        data = poly.fit_transform(line.reshape(1, -1))
        action = trader.predict_action(data.ravel())

        output_file.write(action + "\n")

