# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd
import numpy as np
import sys

''' Loads from sklearn '''
from sklearn.model_selection import ParameterGrid
from IPython.display import clear_output, Markdown, display
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

''' Loads from my algorithms '''
from dif import Dif

sys.path.append('../../metrics/')
from classification_fairness_measures import MeasuresFairness, get_performance_measure_names, get_all_measure_names
from classification_validation import NFold, _StratifiedBy

''' information used from classifiers '''
clf_columns = ['clf_name', 'clf_type', 'clf_params']
dif_columns = ['DIF_1PL', 'DIF_2PL', 'DIF_3PL']

def amount_of_classifiers(classifier_settings):
    ''' Return the amount of classifiers from hyperparameters dict
    '''
    n_classifiers = 0
    for _, (_, param_grid) in classifier_settings.items():
        grid = ParameterGrid(param_grid)
        for _ in grid:
            n_classifiers += 1
    return n_classifiers

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

class NFold_to_DIFModelSelection(NFold):
    ''' Class that runs nfold and calculates the area DIF of classifiers 
    '''
    def __init__(self, classifier_settings, protected_attribute, priv_group, n, print_display):
        super().__init__(n, True, 'group_target', 'fairness_performance', False, None)
        ''' Settings '''
        self.classifier_settings = classifier_settings
        self.priv_group = priv_group
        self.protected_attribute = protected_attribute
        self.print_display = print_display
        self.scores = None
        self.folds_scores = None
        self.current_fold_scores = None
        self.counter = 0
        self.n_classifiers = amount_of_classifiers(self.classifier_settings)
        self.kf = StratifiedKFold(n_splits=n, shuffle=self.shuffle, random_state=self.random_state)
        
    def _params_dif(self, model):
        return {'group' : 'group', 'focal': self.priv_group, 'model': model}

    def _initialize_report(self):
        self.folds_scores = pd.DataFrame(columns=clf_columns + get_all_measure_names() + dif_columns)
    
    def _intialize_current_fold(self):
        self.current_fold_scores = pd.DataFrame(columns=clf_columns + get_all_measure_names() + dif_columns)
    
    def _update_current_fold_report(self, clf_type, params):
        clf_name = clf_type + '_' + str(self.counter)
        info = {'clf_name' : clf_name, 'clf_type' : clf_type, 'clf_params' : str(params)}
        self.current_fold_scores.loc[self.counter] = {**info, **self.measures.scores.iloc[0]}
    
    def _update_reports(self):
        self.folds_scores = pd.concat([self.folds_scores, self.current_fold_scores], ignore_index=True)
    
    def _finish_reports(self):
        by = ['clf_name', 'clf_type', 'clf_params']
        self.folds_scores[get_all_measure_names() + dif_columns] = self.folds_scores[get_all_measure_names() + dif_columns].astype('float64')
        ''' Calculate the average of the folds '''
        self.scores = self.folds_scores.groupby(by=by).mean()
        self.scores = self.scores.reset_index()
        self.scores.index = list(self.scores['clf_name'])
        
    def progress_display(self, clf_name, i, j):
        if self.print_display:
            clear_output()
            print('Validation | Fold %d/%d | Classifier %d/%d (%s)' % 
                  (i, self.n, j, self.n_classifiers, clf_name))
            
    def calculate_DIF(self, item_modeling):
        print('calculating DIF')
        # acertar isso aqui, mudança rápida para os testes
        for model, model_col in zip(['Rasch', '2PL', '3PL'], ['DIF_1PL', 'DIF_2PL', 'DIF_3PL']):
                params = self._params_dif(model)
                methodDif = Dif(params)
                methodDif.calculate(item_modeling, model)
                self.current_fold_scores[model_col] = list(methodDif.dif['DIF'])
        
    def calculate(self, x, y):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(x, y)
        for i, (train_index, test_index) in enumerate(self.kf.split(x, by)):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            '''Run all classifiers '''
            self.counter = 0 # indicate the 'name' of current classifier
            item_modeling = pd.DataFrame() # initialize a new item modeling
            self._intialize_current_fold()
            for clf_type, (Classifier, param_grid) in self.classifier_settings.items():
                grid = ParameterGrid(param_grid)
                for params in grid:
                    self.progress_display(clf_type, i, self.counter)
                    clf =  Classifier(**params)
                    aux_y_train = pd.Series(y_train, index=x_train.index) # make compatible with aif360.
                    clf.fit(x_train, aux_y_train)
                    y_predict = clf.predict(x_test)
                    ''' Calculate thew performance and fairness measures '''
                    self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
                    ''' Update fold report '''
                    self._update_current_fold_report(clf_type, params)
                    ''' Update item modeling '''
                    item_modeling[clf_type + '_' + str(self.counter)] = is_correct(y_test, y_predict)
                    ''' Update counter '''
                    self.counter += 1
            ''' Calculate the area dif of classifiers '''
            item_modeling['group'] = list(x_test.index)
            self.calculate_DIF(item_modeling)
            ''' Update report '''
            self._update_reports()
        self._finish_reports()                      
        
class DIFModelSelection:
    ''' Holdout - separate into validation and training (saves the results), and calculate the area DIF of classifier in 
        cross validation.
    '''
    def __init__(self, classifier_settings, protected_attribute, priv_group, n=10, test_size=0.20, random_state=None, 
                 experiment_name='report', print_reports=False, print_display=True):
        self.classifier_settings = classifier_settings
        self.protected_attribute = protected_attribute
        self.priv_group = priv_group
        self.n = n
        self.test_size = test_size
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.print_reports = print_reports
        self.print_display = print_display
        self.stratified = True
        self.stratified_by = 'group_target'
        self.scores_test = None
        self.scores_validation = None
        self.n_classifiers = amount_of_classifiers(self.classifier_settings)
        self.measures = MeasuresFairness()
    
    def _initialize_reports(self):
        self.scores_test = pd.DataFrame(columns=clf_columns + get_all_measure_names())
        
    def _update_reports(self, clf_type, params):
        info = {'clf_name' : clf_type + '_' + str(self.counter), 'clf_type' : clf_type, 'clf_params' : str(params)}
        self.scores_test.loc[self.counter] = {**info, **self.measures.scores.iloc[0]}
            
    def save_reports(self):  
        if not self.print_reports:
            return None
        self.scores_validation.to_csv(self.experiment_name + '_validation.csv', sep=';', index=False)
        self.scores_test.to_csv(self.experiment_name + '_test.csv', sep=';', index=False)
        
    def progress_display(self, clf_name, i):
        if self.print_display:
            clear_output()
            print('Teste | Classifier %d/%d (%s)' % (i, self.n_classifiers, clf_name))
    
    def calculate(self, X, y):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(X, y)
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=by, test_size=self.test_size, 
                                                            random_state=self.random_state)
        nfold = NFold_to_DIFModelSelection(self.classifier_settings, self.protected_attribute, self.priv_group, 
                                           self.n, self.print_display)
        nfold.calculate(x_train, y_train)
        self.scores_validation = nfold.scores
        self.counter = 0
        ''' retrains all classifiers with the training set and evaluates on the test set '''
        for clf_type, (Classifier, param_grid) in self.classifier_settings.items():
            grid = ParameterGrid(param_grid)
            for params in grid:
                self.progress_display(clf_type, self.counter)
                clf =  Classifier(**params)
                aux_y_train = pd.Series(y_train, index=x_train.index) # make compatible with aif360.
                clf.fit(x_train, aux_y_train)
                y_predict = clf.predict(x_test)
                self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
                self._update_reports(clf_type, params)
                self.counter += 1
        self.scores_test.index = self.scores_test['clf_name']
        self.save_reports()
            
    