import numpy as np
import pandas as pd
import os
import earthpy as et
import math
import statistics
from statistics import stdev
from data_prep2 import Data_prep2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class project_models:
# Code based from Christine Burt's work

    def train_bayes_model(data):
        x=data[:,:-1]#all but last column
        y=data[:,-1]#last column
        #split the data:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
        bayes_model = GaussianNB()
        bayes_model.fit(x_train, y_train)
        
        return bayes_model

# Code based from Steven Portley's work
    def train_logistic_regression(data):
        x=data[:,:-1]#all but last column
        y=data[:,-1]#last column
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

        model = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
        
        return model


# code based from Sue Gerace's work
    def train_decision_tree(data):
        x=data[:,:-1]
        y=data[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
        
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(x_train, y_train)

        return decision_tree_model
    
