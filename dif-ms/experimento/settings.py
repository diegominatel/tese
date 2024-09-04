import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from aif360.sklearn.inprocessing import AdversarialDebiasing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def set_configs(n_columns):
    
    ada = {
        'ADA' : [AdaBoostClassifier,
                {'n_estimators' : list(range(20, 580, 18)), 
                 'random_state' : [42]}]
    }
    
    adversarial_debiasing = {
        'AD' : [AdversarialDebiasing,
                {'prot_attr' : ['Group'],
                 'num_epochs' : list(range(50, 330, 9)),
                 'random_state' : [42]}]
    }
    
    decision_tree = {
        'DT' : [DecisionTreeClassifier,
                {'criterion' : ['gini'],
                 'min_samples_leaf' : list(range(1, 33, 2)),
                 'min_samples_split' : [4, 5],
                 'random_state' : [42]}]
    }
    
    knn = {
        'KNN' : [KNeighborsClassifier,
                 {'n_neighbors' : list(range(1, 17, 2)),
                  'p' : [1, 2, 3, 4], 
                  'n_jobs' : [-1]}]
    }
    
    mlp = {
        'MLP' : [MLPClassifier,
                 {'hidden_layer_sizes' : list(range(10, 74, 2)),
                  'random_state' : [42]}]
    }
    
    rf = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(35, 515, 15)),
                 'min_samples_split' : [math.floor(abs(math.sqrt(n_columns - 1)))], 
                 'random_state' : [42],
                 'n_jobs' : [-1]}]
    }
        
    svm = {
        'SVM' : [SVC,
                 {'kernel' : ['rbf'], 'C' : [0.98, 0.99, 1.00, 1.01], 'gamma' : list(np.arange(0.0025, 0.02, 0.0025)), 
                  'random_state' : [42]}]
    }
    
    xgb = {
        'XGB' : [XGBClassifier,
                 {'n_estimators' : list(range(35, 515, 15)),
                  'random_state' : [42]}]
    }
                                                                 
    
    all_configs = {
        'ad'     : adversarial_debiasing,
        'ada'    : ada,
        'dt'     : decision_tree,
        'knn'    : knn,
        'mlp'    : mlp,
        'rf'     : rf,
        'svm'    : svm,
        'xgb'    : xgb
    }
    
    return all_configs