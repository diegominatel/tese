# -*- coding: utf-8 -*-

''' Bibliotecas gerais '''
import pandas as pd
import numpy as np
import sys
from datetime import datetime

''' Métodos especifico para a aIF 360 '''
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta, RejectOptionClassifier

''' Métodos auxiliares para a organização dos experimentos '''
from sklearn.model_selection import ParameterGrid
from IPython.display import clear_output, display
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from dif_postp import DIF_PostP

sys.path.append('../../metrics/')
from classification_fairness_measures import get_performance_measure_names, get_all_measure_names
from classification_validation import NFold, _StratifiedBy

DIF_methods = [['difArea', '2PL']]
nfold_columns = ['postp', 'method', 'model', 'valid_size', 'threshold']
experiment_columns = ['clf_name', 'clf_type', 'clf_params', 'postp', 'method', 'model', 'valid_size', 'threshold']

class NFold_PostProcessing(NFold):
    ''' Validação de modelos de classificação por nfold (k-fold)
    '''
    def __init__(self, n=5, protected_attribute='Group', priv_group='Privileged', random_state=42, 
                 validation_size=0.25, test_size=0.25):
        
        super().__init__(n, True, 'group_target', 'fairness_performance', False, None)
        ''' Parâmetros exclusivos NFold '''
        self.priv_group = priv_group
        self.protected_attribute = protected_attribute
        self.validation_size = validation_size
        self.test_size = test_size
        self.validation_scores = None
        self.folds_scores = None
        self.folds_groupby_scores = None
        self.random_state = random_state
        ''' Seleciona o método de NFold '''
        Fold = StratifiedKFold
        ''' Instância o método de KFold de acordo com os parâmetros '''
        self.kf = Fold(n_splits=n, shuffle=False, random_state=None)
        
    def _initialize_reports(self):
        ''' Método que inicia o relatório no padrão PostP Analysis
        '''
        self.test_scores = pd.DataFrame(columns=nfold_columns + get_all_measure_names())
        self.folds_scores = pd.DataFrame(columns=nfold_columns + get_all_measure_names())
        self.folds_groupby_scores = pd.DataFrame(columns=nfold_columns + ['group'] + get_performance_measure_names())
        
    def info_score(self, postp, method, valid_size, ppa):
        score = pd.DataFrame(columns=nfold_columns + get_all_measure_names())
        if postp == 'dif-postp':
            info = {'postp': postp}
            score.loc[0] = {**info, **ppa, **self.measures.scores.iloc[0]}
            return score
        else:
            info = {'postp':postp, 'method':method, 'valid_size': str(valid_size), 
                    'model':'-', 'threshold':'-'}
            score.loc[0] = {**info, **self.measures.scores.iloc[0]}
            return score
        
    def info_group_score(self, postp, method, valid_size, ppa, j):
        score = pd.DataFrame(columns=nfold_columns + ['group'] + get_performance_measure_names())
        if postp == 'dif-postp':
            info = {'postp': postp}
            score.loc[0] = {**info, **ppa, **self.measures.groupby_scores.iloc[j]}
            return score
        else:
            info = {'postp':postp, 'method':method, 'valid_size':str(valid_size), 
                    'model':'-', 'threshold':'-'}
            score.loc[0] = {**info, **self.measures.groupby_scores.iloc[j]}
            return score
    
    def _update_validation_reports(self, postp, method='-', valid_size='-', ppa=None):
        ''' Atualiza os resultados no relatório
        '''
        scores = self.info_score(postp, method, valid_size, ppa)
        self.folds_scores = pd.concat([self.folds_scores, scores], ignore_index=True)
        
        for j in range(self.measures.groupby_scores.shape[0]):
            group_scores = self.info_group_score(postp, method, valid_size, ppa, j)
            self.folds_groupby_scores = pd.concat([self.folds_groupby_scores, group_scores], ignore_index=True)
            
    def _update_test_reports(self, postp, method='-', valid_size='-', ppa=None):
        scores = self.info_score(postp, method, valid_size, ppa)
        self.test_scores = pd.concat([self.test_scores, scores], ignore_index=True)
        
    def _finish_validation_reports(self):
        ''' Força os atributos numéricos a serem numéricos '''
        by = ['postp', 'method', 'model', 'valid_size']
        self.folds_scores[get_all_measure_names()] = self.folds_scores[get_all_measure_names()].astype('float64')
        self.validation_scores = self.folds_scores.groupby(by=by).mean()
        self.validation_scores = self.validation_scores.reset_index()
        self.groupby_scores = self.folds_groupby_scores.groupby(by=by + ['group']).mean()
        self.groupby_scores = self.groupby_scores.reset_index()
        
    
    def execute_without(self, clf, x_train, x_test, y_train, y_test, execution_type):
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
        
        if execution_type == 'validation':
            self._update_validation_reports('without')
        else:
            self._update_test_reports('without')
        
    def execute_ceo(self, clf, x_train, x_test, y_train, y_test, execution_type):
        
        for cost in ['weighted', 'fpr', 'fnr']:
            
            ceo = CalibratedEqualizedOdds(self.protected_attribute, cost_constraint=cost)
            post = PostProcessingMeta(estimator=clf, postprocessor=ceo, val_size=self.validation_size, prefit=False)
            aux_y_train = pd.Series(y_train, index=x_train.index) # ajusta pra ficar compativel
            post.fit(x_train, aux_y_train)
            y_predict = post.predict(x_test)
            self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
            
            if execution_type == 'validation':
                self._update_validation_reports('ceo', cost, self.validation_size)
            else:
                self._update_test_reports('ceo', cost, self.validation_size)
        
    def execute_roc(self, clf, x_train, x_test, y_train, y_test, execution_type):
        
        for threshold in np.arange(0.35, 0.65, 0.025):
            roc = RejectOptionClassifier(prot_attr= self.protected_attribute, threshold=threshold)
            post = PostProcessingMeta(estimator=clf, postprocessor=roc, val_size=self.validation_size, prefit=False)
            aux_y_train = pd.Series(y_train, index=x_train.index) # ajusta pra ficar compativel
            post.fit(x_train, aux_y_train, priv_group=self.priv_group)
            y_predict = post.predict(x_test)
            self.measures.calculate(y_test, y_predict, x_test.index, self.priv_group)
            
            if execution_type == 'validation':
                self._update_validation_reports('roc', threshold, self.validation_size)
            else:
                self._update_test_reports('roc', threshold, self.validation_size)
                    
    
    def execute_difpostp(self, clf, x_train, x_test, y_train, y_test, execution_type):
        
        ppa = DIF_PostP(clf, valid_size=self.validation_size, focal=self.priv_group, DIF_methods=DIF_methods,
                        random_state=self.random_state)
        ppa.fit(x_train, y_train)
        ''' Realiza as predições '''
        y_predicts = ppa.predict(x_test)
        ''' Calcula as métricas (métricas diferentes, chamadas diferentes -- funciona só para fairness_performance) '''
        for idx in y_predicts.columns:
            self.measures.calculate(y_test, y_predicts[idx].ravel(), x_test.index, self.priv_group)
            ''' Atualiza o relatório '''
            if execution_type == 'validation':
                self._update_validation_reports('dif-postp', ppa = ppa.thresholds.loc[idx])
            else:
                self._update_test_reports('dif-postp', ppa = ppa.thresholds.loc[idx])
                    
    def nfold(self, x, y, clf):
        ''' Estratifica para validação e teste da validação'''
        ''' Se for estratificado retorna o vetor com as classes de estratificação '''
        by = getattr(_StratifiedBy, self.stratified_by)(x, y)
        ''' Realiza o k-fold '''
        for i, (train_index, test_index) in enumerate(self.kf.split(x, by)):
            ''' Separa os conjuntos de treino e teste '''
            print('[%s] - Initialize Fold %d/%d)' % (datetime.now(), i, self.n))
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ''' Executa sem pós-processamento '''
            print('[%s] - Initialize Validation (without))' % (datetime.now()))
            self.execute_without(clf, x_train, x_test, y_train, y_test, 'validation')
            ''' Executa Calibrated Equalized Odds '''
            print('[%s] - Initialize Validation (CEO))' % (datetime.now()))
            self.execute_ceo(clf, x_train, x_test, y_train, y_test, 'validation')
            ''' Executa Reject Option '''
            print('[%s] - Initialize Validation (ROC))' % (datetime.now()))
            self.execute_roc(clf, x_train, x_test, y_train, y_test, 'validation')
            ''' Executa DIF-PostP '''
            print('[%s] - Initialize Validation (DIF-PostP))' % (datetime.now()))
            self.execute_difpostp(clf, x_train, x_test, y_train, y_test, 'validation')
    
    def calculate(self, x, y, clf):
        ''' Inicializa os relatórios (para não agregar) '''
        print('[%s] - Initialize reports' % (datetime.now()))
        self._initialize_reports()
        ''' Realiza NFold (Validação) '''
        self.nfold(x, y, clf)
        print('[%s] - Update Reports)' % (datetime.now()))
        self._finish_validation_reports()

class PerformExperiment_PostProcessing():
    ''' Classe que realiza um conjunto de experimentos e gera relatórios (exclusivo uso para PPA)
    '''
    def __init__(self, classifier_settings, protected_attribute, priv_group, 
                 validation_size=0.25, test_size=0.25, n=5, experiment_name='report', print_reports=False, 
                 print_display=True, random_state=42):
        
        self.classifier_settings = classifier_settings
        self.protected_attribute = protected_attribute
        self.priv_group = priv_group
        self.experiment_name = experiment_name
        self.print_reports = print_reports
        self.print_display = print_display
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state   
        self.n = n
        self.validation = NFold_PostProcessing(protected_attribute=protected_attribute, priv_group=priv_group, 
                                               validation_size = self.validation_size, n = self.n,
                                               test_size = self.test_size, random_state = self.random_state)
        ''' Parâmetros internos '''
        self.counter = 0
        self.n_classifiers = self.get_number_classifiers()
        self.validation_scores = None
        self.test_scores = None
        self.groupby_scores = None
    
    def get_number_classifiers(self):
        ''' Recupera o número de classificadores no experimento para auxiliar no progresso do experimento
        '''
        n_classifiers = 0
        for _, (_, param_grid) in self.classifier_settings.items():
            grid = ParameterGrid(param_grid)
            for _ in grid:
                n_classifiers += 1
        return n_classifiers
    
    def _display(self, clf_name):
        ''' Função que mostra o progresso no jupyter
        Args:
            clf_name (str): nome do classificador
        '''
        if self.print_display:
            clear_output()
            print('%d/%d - (%s)' % (self.counter, self.n_classifiers, clf_name))
    
    def _initialize_reports(self):
        self.counter = 0
        self.validation_scores = pd.DataFrame(columns=experiment_columns + get_all_measure_names())
        self.test_scores = pd.DataFrame(columns=experiment_columns + get_all_measure_names())
        self.groupby_scores = pd.DataFrame(columns=experiment_columns + ['group'] + get_performance_measure_names())
        
    def _update_reports(self, clf_type, params):
        ''' Atualiza os relatórios
        Args:
            clf_name (str): nome do classificador
            params (dict): lista de parâmetros
        '''
        ''' Concatena o nome do classificador com a ordem de execução '''
        clf_name = clf_type + '_' + str(self.counter)
        ''' Recupera os resultados '''
        validation_scores = self.validation.validation_scores
        test_scores = self.validation.test_scores
        ''' Copia as informações do classificador '''
        n = self.validation.validation_scores.shape[0]
        validation_scores['clf_name'], validation_scores['clf_type'], validation_scores['clf_params'] = ([clf_name]*n,
                                                                                                        [clf_type]*n,
                                                                                                        [str(params)]*n)
        test_scores['clf_name'], test_scores['clf_type'], test_scores['clf_params'] = ([clf_name]*n, [clf_type]*n,
                                                                                       [str(params)]*n)
        ''' Concatena os resultados no relatório final '''
        self.validation_scores = pd.concat([self.validation_scores, validation_scores], ignore_index=True)
        self.test_scores = pd.concat([self.test_scores, test_scores], ignore_index=True)
        
      
        ''' Por grupo '''
        group_scores = self.validation.groupby_scores
        n = self.validation.groupby_scores.shape[0]
        group_scores['clf_name'], group_scores['clf_type'], group_scores['clf_params'] = [clf_name]*n, [clf_type]*n, [str(params)]*n
        ''' Concatena os resultados no relatório final '''
        self.groupby_scores = pd.concat([self.groupby_scores, group_scores], ignore_index=True)
        
    
    def _print_reports(self):
        self.validation_scores.to_csv(self.experiment_name + '_validation.csv', sep=';', index=False)
        self.test_scores.to_csv(self.experiment_name + '_test.csv', sep=';', index=False)
        
    
    def calculate(self, x, y):
        ''' Realiza o experimento
        Args:
            x (pd.DataFrame): conjunto de dados
            y (array): classes
        '''
        ''' Inicializa o relatório '''
        self._initialize_reports()
        ''' Realiza o experimento para todos os diferentes algoritmos de classificações e parâmetros '''
        for clf_type, (Classifier, param_grid) in self.classifier_settings.items():
            grid = ParameterGrid(param_grid)
            for params in grid:
                self._display(clf_type)
                ''' realiza o método de validação selecionado '''
                clf =  Classifier(**params)
                self.validation.calculate(x, y, clf)
                ''' atualiza os resultados nos relatórios '''
                self._update_reports(clf_type, params)
                ''' atualiza o contador com o número de classificadores testados '''
                self.counter += 1
        ''' limpa os outputs '''
        if self.print_display:
            clear_output()
        if self.print_reports:
            self._print_reports()
    
    