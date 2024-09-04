# -*- coding: utf-8 -*-

''' Bibliotecas gerais '''
import pandas as pd
import numpy as np
import sys
from datetime import datetime

''' Bibliotecas para usar o DIF '''
from dif import Dif
from sklearn.model_selection import train_test_split

''' Métodos auxiliares para a organização dos experimentos '''
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve

sys.path.append('../../metrics/')
from classification_validation import _StratifiedBy

report_columns = ['method', 'model', 'valid_size', 'threshold']
n_items = [16, 32]
per = [0.05, 0.10, 0.20, 0.40, 0.45]

def is_correct(true, predict):
    ''' Função que retorna os exemplos acertados por uma predição (vetorizado posteriormente)
    Args:
        true: (list) classe verdadeira
        predict: (list) classe predita pelo classificador

    Returns: (list) que o valor 1 indica acerto no exemplo e 0 errado
    '''
    if true == predict:
        return 1
    else:
        return 0    

def readjust_predict(predict_proba, positive_value):
    if predict_proba >= positive_value:
        return 1
    else:
        return 0    
    
''' vetoriza as funções '''    
is_correct = np.vectorize(is_correct)
readjust_predict = np.vectorize(readjust_predict)
    

# refatorar esse código
def repeat_columns(data):
    ''' Elimina as colunas com o mesmos valores em todas as linhas, conserva a primeira ocorrência da coluna repetida
    Args:
        data: (pd.DataFrame)
    Returns:
    '''
    drop_columns = []
    columns = data.columns
    i = 0
    while i < len(data.columns) - 1:
        if columns[i] not in drop_columns:
            j = i + 1
            while j < len(data.columns):
                if pd.DataFrame.equals(data[columns[i]], data[columns[j]]) == True:
                    drop_columns.append(columns[j])
                j += 1
        i += 1
    return drop_columns

# refatorar esse código
def only_one_response(data):
    ''' Elimina colunas que tem uma única resposta (certo ou errado) em alguns dos grupos '''
    drop_columns = []
    for group in list(data['group'].unique()):
        aux_data = data.loc[data['group'] == group]
        for col in aux_data.columns:
            if col != 'group':
                if len(aux_data[col].unique()) == 1:
                    if col not in drop_columns:
                        drop_columns.append(col)
    
    return drop_columns

class DIF_PostP(BaseEstimator, ClassifierMixin):
    ''' Classe que executa o PostP-DIF para todos os métodos de DIF e para todas as combinações de seleção de threshold '''
    
    def __init__(self, clf, focal, valid_size, DIF_methods, random_state):
        ''' '''
        self.clf = clf
        self.focal = focal
        self.valid_size = valid_size
        self.focal = focal
        self.DIF_methods = DIF_methods
        self.thresholds = None
        self.ppa_counter = 0
        self.random_state = random_state
        
    def get_params(self, model):
        return {'group' : 'group', 'focal' : self.focal, 'model' : model}
    
    def _initialize_report(self):
        self.thresholds = pd.DataFrame(columns=report_columns)
    
    def update_except_thresholds(self, method_name, model, threshold):
        self.thresholds.loc[self.ppa_counter] = [method_name, model, self.valid_size, threshold]
        self.ppa_counter += 1
            
    def update_thresholds(self, method_name, model):
        threshold = float(self.validation_values['DIF'].idxmin())
        self.thresholds.loc[self.ppa_counter] = [method_name, model, self.valid_size, threshold]
        self.ppa_counter += 1
        
    def generate_item_modeling(self, x_test_valid, y_test_valid, threshold_items):
        '''
        '''
        item_modeling =  pd.DataFrame(columns=threshold_items)
        for i, value in enumerate(threshold_items):
            predict = readjust_predict(self.proba_positive, value)
            item_modeling[value] = is_correct(y_test_valid, predict)
        ''' deleta as colunas repetidas '''    
        r_columns = repeat_columns(item_modeling)
        item_modeling = item_modeling.drop(r_columns, axis=1)
        ''' insere a coluna grupo '''
        item_modeling['group'] = list(x_test_valid.index)
        ''' arruma o validation '''
        self.validation_values = self.validation_values.drop(r_columns, axis=0)
        
        ''' elimina colunas (desagrupadas) que tem a mesma resposta para todas as instâncias '''
        drop = only_one_response(item_modeling)
        if len(drop) > 0:
            item_modeling = item_modeling.drop(drop, axis=1)
            self.validation_values = self.validation_values.drop(drop, axis=0)
            
        return item_modeling
    
    
    def fit(self, x, y):
        ''' Inicializa relatório '''
        self._initialize_report()
        ''' Estratificação por grupo protegido e classe '''
        by = getattr(_StratifiedBy, 'group_target')(x, y)
        x_train_valid, x_test_valid, y_train_valid, y_test_valid = train_test_split(x, y, stratify=by, 
                                                                                    test_size=self.valid_size,
                                                                                    random_state=self.random_state)
        ''' Treina o classificador '''
        print('-------------------------------------------------------------')
        print('[%s] - Treina o classificador para validação' % datetime.now())
        self.clf.fit(x_train_valid, y_train_valid)
       
        print('[%s] - Classificador treinado para validação' % datetime.now())
        ''' Realiza a predição (recupera a probabilide das classes) '''
        proba = self.clf.predict_proba(x_test_valid)
        ''' Somente a probabilidade de classificar na classe positiva '''
        self.proba_positive = np.transpose(proba)[1]  
        _, _, all_thresholds = roc_curve(y_test_valid, self.proba_positive)
        
        ''' somente para ajustar quando não executa '''
        percents = pd.DataFrame()
        percents['Group'] = x.index
        percents['Target'] = y
        percents = percents.value_counts()

        for p in per:
            for N in n_items:
                select_thresholds = all_thresholds[round(len(all_thresholds)*p):-(round(len(all_thresholds)*p))]
                threshold_items = np.sort(select_thresholds)
                if len(threshold_items) > N:
                    threshold_items = [threshold_items[i] for i in np.linspace(0.5,len(threshold_items)-0.5, N, dtype=int)]

                self.validation_values = pd.DataFrame(index=threshold_items, columns=report_columns)
                item_modeling = self.generate_item_modeling(x_test_valid, y_test_valid, threshold_items)

                print('[%s] - Gera o item_modelig (%s)' % (datetime.now(), str(item_modeling.shape)))
                ''' Executa todas as combinações de método de seleção e métodos de DIF '''
                for method_name, model in self.DIF_methods:
                    ''' Parametrização para executar o DIF no R '''   
                    params = self.get_params(model)
                    ''' Instancia o método de DIF em questão '''
                    print('[%s] - Executa o DIF (%s)' % (datetime.now(), method_name))
                    methodDif = Dif(params)
                    ''' Executa o método de DIF em questão '''
                    try:
                        ''' Sobe exceção se o item modeling não estiver no padrão minimo desejado '''
                        if item_modeling.shape[1] < 5:
                            raise ValueError('Item modeling: número de colunas insuficientes')
                        if len(item_modeling.loc[item_modeling['group'] == self.focal]) < 5:
                            raise ValueError('Item modeling: número de linhas grupo focal insuficientes')
                        if len(item_modeling.loc[item_modeling['group'] != self.focal]) < 5:
                            raise ValueError('Item modeling: número de linhas grupo não-focal insuficientes')
                        methodDif.calculate(item_modeling, model)
                        ''' Atualiza a matriz de validação '''
                        self.validation_values['DIF'] = methodDif.dif['DIF'].astype(float)
                        ''' Atualiza os thresholds para todos os métodos de seleção '''
                        self.update_thresholds(model + '_N-' + str(N) + '_P-' + str(p), method_name)
                    except:
                        ''' Caso de erro na execução atualiza todos os thresholds '''
                        middle = (percents['Privileged', 0] + percents['Unprivileged', 0])/len(x)
                        self.update_except_thresholds(model + '_N-' + str(N) + '_P-' + str(p), method_name, middle)
                ''' Retreina o classificador com todos dados '''
                print('[%s] - Treina o Classificador' % (datetime.now()))
                self.clf.fit(x, y)
                print('[%s] - Fim deste processamento' % (datetime.now()))
        
    def predict(self, x):
        proba = self.clf.predict_proba(x)
        ''' Pega somente a probabilidade de classificar como positivo '''
        proba_positive = np.transpose(proba)[1]
        ''' Conjunto de predições '''
        predicts = pd.DataFrame(columns=list(self.thresholds.index))
        for idx in self.thresholds.index:
            predicts[idx] = readjust_predict(proba_positive, self.thresholds.loc[idx, 'threshold'])
            
        return predicts