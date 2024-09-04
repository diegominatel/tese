# -*- coding: utf-8 -*-

''' General packages '''
import math
import numpy as np
import pandas as pd

''' Load classification algorithms form sklearn '''
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

''' class Mirt from mirt.py '''
from mirt import Mirt

def is_correct(true, predict):
    ''' Check correct and incorrect predictions
    Args:
        true: (list) true labels
        predict: (list) predicted target values
    Returns: (list) indicates correct and incorrect predictions
    '''
    if true == predict:
        return 1
    else:
        return 0     
    
is_correct = np.vectorize(is_correct)  


class Preprocessing_DIF_DT():
    ''' Estimates the parameters of the items so that there is no DIF. Thus, using the abilities as a weight in the training of         classifiers
    ''' 
    def __init__(self):
        self.weights = None
        
        
    def execute(self, X, y):
        ''' Training the set of classifiers '''
        dt1 = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split = 5, random_state=42).fit(X, y)
        dt2 = DecisionTreeClassifier(min_samples_leaf=12, min_samples_split = 5, random_state=42).fit(X, y)
        dt3 = DecisionTreeClassifier(min_samples_leaf=22, min_samples_split = 5, random_state=42).fit(X, y)
        ''' Item modeling '''
        pred_dt1 = is_correct(y, dt1.predict(X))
        pred_dt2 = is_correct(y, dt2.predict(X))
        pred_dt3 = is_correct(y, dt3.predict(X))
        item_modeling = pd.DataFrame(np.transpose([pred_dt1, pred_dt2, pred_dt3]))
        ''' Insert the groups in the item modeling '''
        group = list(X.index)
        ''' Insert the condition that all classifier have one incorrect and correct answer in both groups 
            (necessary to always run) '''
        group.append('Privileged')
        group.append('Privileged')
        group.append('Unprivileged')
        group.append('Unprivileged')
        item_modeling.loc[item_modeling.shape[0]] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 1] = [1]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 2] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 3] = [1]*item_modeling.shape[1]
        ''' Run Mirt '''
        mirt_model = Mirt()
        mirt_model.calculate(item_modeling, group)
        weights = mirt_model.estimated_abilities
        weights = weights*(-1)
        ''' remove the weights of insertion of line 56-63 '''
        n = len(X)
        weights = weights.drop([n, n + 1, n + 2, n + 3], axis=0)
        ''' Rescale weights ''' 
        return (MinMaxScaler(feature_range=(1, 8)).fit_transform(weights))
    
class Preprocessing_DIF_KNN():
    ''' Estimates the parameters of the items so that there is no DIF. Thus, using the abilities as a weight in the training of         classifiers
    ''' 
    def __init__(self):
        self.weights = None
        
    def execute(self, X, y):
        ''' Training the set of classifiers '''
        knn1 = KNeighborsClassifier(n_neighbors = 3, n_jobs=-1).fit(X, y)
        knn2 = KNeighborsClassifier(n_neighbors = 11, n_jobs=-1).fit(X, y)
        knn3 = KNeighborsClassifier(n_neighbors = 23, n_jobs=-1).fit(X, y)
        ''' Item modeling '''
        pred_knn1 = is_correct(y, knn1.predict(X))
        pred_knn2 = is_correct(y, knn2.predict(X))
        pred_knn3 = is_correct(y, knn3.predict(X))
        item_modeling = pd.DataFrame(np.transpose([pred_knn1, pred_knn2, pred_knn3]))
        ''' Insert the groups in the item modeling '''
        group = list(X.index)
        ''' Insert the condition that all classifier have one incorrect and correct answer in both groups 
            (necessary to always run) '''
        group.append('Privileged')
        group.append('Privileged')
        group.append('Unprivileged')
        group.append('Unprivileged')
        item_modeling.loc[item_modeling.shape[0]] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 1] = [1]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 2] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 3] = [1]*item_modeling.shape[1]
        ''' Run Mirt '''
        mirt_model = Mirt()
        mirt_model.calculate(item_modeling, group)
        weights = mirt_model.estimated_abilities
        weights = weights*(-1)
        ''' remove the weights of insertion of line 56-63 '''
        n = len(X)
        weights = weights.drop([n, n + 1, n + 2, n + 3], axis=0)
        ''' Rescale weights ''' 
        return (MinMaxScaler(feature_range=(1, 8)).fit_transform(weights))
    
class Preprocessing_DIF_MLP():
    ''' Estimates the parameters of the items so that there is no DIF. Thus, using the abilities as a weight in the training of         classifiers
    ''' 
    def __init__(self):
        self.weights = None
        
        
    def execute(self, X, y):
        ''' Training the set of classifiers '''
        mlp1 = MLPClassifier(hidden_layer_sizes = (5, ), random_state=42).fit(X, y)
        mlp2 = MLPClassifier(hidden_layer_sizes = (10, ), random_state=42).fit(X, y)
        mlp3 = MLPClassifier(hidden_layer_sizes = (15, ), random_state=42).fit(X, y)
        ''' Item modeling '''
        pred_mlp1 = is_correct(y, mlp1.predict(X))
        pred_mlp2 = is_correct(y, mlp2.predict(X))
        pred_mlp3 = is_correct(y, mlp3.predict(X))
        item_modeling = pd.DataFrame(np.transpose([pred_mlp1, pred_mlp2, pred_mlp3]))
        ''' Insert the groups in the item modeling '''
        group = list(X.index)
        ''' Insert the condition that all classifier have one incorrect and correct answer in both groups 
            (necessary to always run) '''
        group.append('Privileged')
        group.append('Privileged')
        group.append('Unprivileged')
        group.append('Unprivileged')
        item_modeling.loc[item_modeling.shape[0]] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 1] = [1]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 2] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 3] = [1]*item_modeling.shape[1]
        ''' Run Mirt '''
        mirt_model = Mirt()
        mirt_model.calculate(item_modeling, group)
        weights = mirt_model.estimated_abilities
        weights = weights*(-1)
        ''' remove the weights of insertion of line 56-63 '''
        n = len(X)
        weights = weights.drop([n, n + 1, n + 2, n + 3], axis=0)
        ''' Rescale weights ''' 
        return (MinMaxScaler(feature_range=(1, 8)).fit_transform(weights))