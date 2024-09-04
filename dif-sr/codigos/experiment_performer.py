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
from aif360.sklearn.preprocessing import Reweighing

''' Loads from my algorithms '''
from preprocessing_DIF import Preprocessing_DIF_KNN, Preprocessing_DIF_DT, Preprocessing_DIF_MLP

sys.path.append('../../metrics/')
from classification_fairness_measures import MeasuresFairness, get_performance_measure_names, get_all_measure_names
from classification_validation import NFold, _StratifiedBy

''' information used from classifiers '''
clf_columns = ['clf_name', 'clf_type', 'clf_params', 'preprocessing']

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

class NFold_Preprocessing(NFold):
    ''' Class that runs nfold and preprocessing fairness algorithms  
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
        self.counter = 0
        self.n_classifiers = amount_of_classifiers(self.classifier_settings)
        self.kf = StratifiedKFold(n_splits=n, shuffle=self.shuffle, random_state=self.random_state)

    def _initialize_report(self):
        self.folds_scores = pd.DataFrame(columns=clf_columns + get_all_measure_names())
    
    def _update_report(self, clf_type, params, preprocessing):
        scores = pd.DataFrame(columns=clf_columns + get_all_measure_names())
        clf_name = clf_type + '_' + str(self.counter)
        info = {'clf_name' : clf_name, 'clf_type' : clf_type, 'clf_params' : str(params), 'preprocessing' : preprocessing}
        scores.loc[0] = {**info, **self.measures.scores.iloc[0]}
        self.folds_scores = pd.concat([self.folds_scores, scores], ignore_index=True)
    
    def _finish_reports(self):
        by = ['clf_name', 'clf_type', 'clf_params', 'preprocessing']
        self.folds_scores[get_all_measure_names()] = self.folds_scores[get_all_measure_names()].astype('float64')
        ''' Calculate the average of the folds '''
        self.scores = self.folds_scores.groupby(by=by).mean()
        self.scores = self.scores.reset_index()
        
    def progress_display(self, clf_name, i, j):
        if self.print_display:
            clear_output()
            print('Validation | Fold %d/%d | Classifier %d/%d (%s)' % 
                  (i, self.n, j, self.n_classifiers, clf_name))
    
    def execute_without(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'without')
        
    def execute_reweighing(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run Reweighing '''
        aux_clf = Reweighing(prot_attr='Group')
        weights = aux_clf.fit_transform(x_train, y_train)[1]       
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(weights.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'reweighing')
        
    def execute_dif_dt(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_DT()
        self.weights_dt = aux_clf.execute(x_train, y_train) 
        
    def execute_preprocessing_dif_dt(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''          
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_dt.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_dt')
    
    def execute_dif_knn(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_KNN()
        self.weights_knn = aux_clf.execute(x_train, y_train) 
        
    def execute_preprocessing_dif_knn(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''         
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_knn.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_knn')
        
    def execute_dif_mlp(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_MLP()
        self.weights_mlp = aux_clf.execute(x_train, y_train) 
    
    def execute_preprocessing_dif_mlp(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''        
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_mlp.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_mlp')
    
    def calculate(self, x, y):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(x, y)
        for i, (train_index, test_index) in enumerate(self.kf.split(x, by)):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.execute_dif_dt(x_train, y_train)
            self.execute_dif_knn(x_train, y_train)
            self.execute_dif_mlp(x_train, y_train)
            '''Run all classifiers '''
            self.counter = 0 # indicate the 'name' of current classifier
            for clf_type, (Classifier, param_grid) in self.classifier_settings.items():
                grid = ParameterGrid(param_grid)
                for params in grid:
                    self.progress_display(clf_type, i, self.counter)
                    clf =  Classifier(**params)
                    aux_y_train = pd.Series(y_train, index=x_train.index) # make compatible with aif360.
                    ''' Run without insert weights on the examples '''
                    self.execute_without(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                    ''' Run with reweighing algorithm '''
                    self.execute_reweighing(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                    ''' Run with preprocessing dif dt '''
                    self.execute_preprocessing_dif_dt(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                    ''' Run with preprocessing dif knn '''
                    self.execute_preprocessing_dif_knn(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                    ''' Run with preprocessing dif mlp '''
                    self.execute_preprocessing_dif_mlp(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                    ''' Increment the counter '''
                    self.counter += 1
        self._finish_reports()                      
        
class ExperimentPerformer:
    ''' Holdout - separate into validation and training (saves the results), and and preprocessing fairness algorithms  
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
        
    def _update_report(self, clf_type, params, preprocessing):
        scores = pd.DataFrame(columns=clf_columns + get_all_measure_names())
        clf_name = clf_type + '_' + str(self.counter)
        info = {'clf_name' : clf_name, 'clf_type' : clf_type, 'clf_params' : str(params), 'preprocessing' : preprocessing}
        scores.loc[0] = {**info, **self.measures.scores.iloc[0]}
        self.scores_test = pd.concat([self.scores_test, scores], ignore_index=True)
            
    def save_reports(self):  
        if not self.print_reports:
            return None
        self.scores_validation.to_csv(self.experiment_name + '_validation.csv', sep=';', index=False)
        self.scores_test.to_csv(self.experiment_name + '_test.csv', sep=';', index=False)
        
    def progress_display(self, clf_name, i):
        if self.print_display:
            clear_output()
            print('Teste | Classifier %d/%d (%s)' % (i, self.n_classifiers, clf_name))
                  
    def execute_without(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'without')
        
    def execute_reweighing(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run Reweighing '''
        aux_clf = Reweighing(prot_attr='Group')
        weights = aux_clf.fit_transform(x_train, y_train)[1]       
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(weights.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'reweighing')
        
    def execute_dif_dt(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_DT()
        self.weights_dt = aux_clf.execute(x_train, y_train)
        
    def execute_preprocessing_dif_dt(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''      
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_dt.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_dt')
        
    def execute_dif_knn(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_KNN()
        self.weights_knn = aux_clf.execute(x_train, y_train)
        
    def execute_preprocessing_dif_knn(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''        
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_knn.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_knn')
        
    def execute_dif_mlp(self, x_train, y_train):
        ''' Run preprocessing dif '''
        aux_clf = Preprocessing_DIF_MLP()
        self.weights_mlp = aux_clf.execute(x_train, y_train) 
        
    def execute_preprocessing_dif_mlp(self, clf, clf_type, params, x_train, x_test, y_train, y_test):
        ''' Run preprocessing dif '''        
        ''' Execute training with weights '''          
        clf.fit(x_train, y_train, list(self.weights_mlp.ravel()))
        y_predict = clf.predict(x_test)
        ''' Calculate the performance and fairness measures '''
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        ''' Update fold report '''
        self._update_report(clf_type, params, 'dif_mlp')
    
    def calculate(self, X, y):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(X, y)
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=by, test_size=self.test_size, 
                                                            random_state=self.random_state)
        nfold = NFold_Preprocessing(self.classifier_settings, self.protected_attribute, self.priv_group, 
                                           self.n, self.print_display)
        nfold.calculate(x_train, y_train)
        self.scores_validation = nfold.scores
        self.counter = 0
        self.execute_dif_dt(x_train, y_train)
        self.execute_dif_knn(x_train, y_train)
        self.execute_dif_mlp(x_train, y_train)
        ''' retrains all classifiers with the training set and evaluates on the test set '''
        for clf_type, (Classifier, param_grid) in self.classifier_settings.items():
            grid = ParameterGrid(param_grid)
            for params in grid:
                self.progress_display(clf_type, self.counter)
                clf =  Classifier(**params)
                aux_y_train = pd.Series(y_train, index=x_train.index) # make compatible with aif360.
                ''' Run without insert weights on the examples '''
                self.execute_without(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                ''' Run with reweighing algorithm '''
                self.execute_reweighing(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                ''' Run with preprocessing dif dt '''
                self.execute_preprocessing_dif_dt(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                ''' Run with preprocessing dif knn '''
                self.execute_preprocessing_dif_knn(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                ''' Run with preprocessing dif mlp '''
                self.execute_preprocessing_dif_mlp(clf, clf_type, params, x_train, x_test, aux_y_train, y_test)
                ''' Increment the counter '''
                self.counter += 1
        self.save_reports()
            
    