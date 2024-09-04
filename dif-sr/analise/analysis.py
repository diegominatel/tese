import pandas as pd
import numpy as np
import math
import sys
import os
from IPython.display import display
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

measures = ['ratio_selection_rate', 'ratio_recall', 'ratio_odds']

datasets = ['adult', 'amphet', 'arrhythmia', 'bank', 'compasmen', 'compaswomen', 'contraceptive', 'ecstasy', 
            'german', 'heart', 'student', 'titanic']

algorithms = ['ada', 'dt', 'rf', 'svm', 'xgb']

def multicriteria(measurement_values):
    pairs = list(combinations(measurement_values, 2))
    area = 0
    for a, b in pairs:
        area += (a*b*math.sin((2*math.pi)/3)/2)    
    return area

def multicriteria_validation(validation):
    matrix = validation[measures].to_numpy()
    validation['multicriteria'] = [multicriteria(row) for row in matrix]
    return validation

def report_best_model(measure):

    results_valid = pd.DataFrame(columns = ['Dataset', 'Config', 'SRA', 'DIF-SR', 'Reweighing'])
    results_test = pd.DataFrame(columns = ['Dataset', 'Config',  'SRA', 'DIF-SR', 'Reweighing'])
    f1 = pd.DataFrame(columns = ['Dataset', 'Config',  'SRA', 'DIF-SR', 'Reweighing'])
    count = 0

    for name in datasets:
        for config in algorithms:
            ''' abre os resultados '''
            if os.path.exists('../experimento/' + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv('../experimento/' + name + '_' + config + '_validation.csv', 
                                         sep=';', index_col=0) 
                test = pd.read_csv('../experimento/' + name + '_' + config + '_test.csv', 
                                   sep=';', index_col=0)
                
                ''' Calculate ratio odds '''
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2
                
                ''' Calculate multicriteria '''
                validation = multicriteria_validation(validation)
                test = multicriteria_validation(test)

                values_test = []
                values_test.append(name)
                values_test.append(config)
                
                values_f1 = []
                values_f1.append(name)
                values_f1.append(config)

                for preprocessing in ['without', 'dif', 'reweighing']:

                    aux_validation = validation.loc[validation['preprocessing'].str.contains(preprocessing)].copy()
                    aux_validation = aux_validation.sort_values(by=['clf_name', 'preprocessing'])
                    aux_validation = aux_validation.reset_index()

                    aux_test = test.loc[test['preprocessing'].str.contains(preprocessing)].copy()
                    aux_test = aux_test.sort_values(by=['clf_name', 'preprocessing'])
                    aux_test = aux_test.reset_index()

                    idx = aux_validation[measure].idxmax()
                    values_test.append(aux_test.loc[idx, measure])
                    values_f1.append(aux_test.loc[idx, 'f1_macro'])
                    
                
                results_test.loc[count] = values_test
                f1.loc[count] = values_f1
                count += 1
                
    return results_test, f1

def report_validation(measure):

    results_valid = pd.DataFrame(columns = ['Dataset', 'Config', 'SRA', 'DIF-SR', 'Reweighing'])
    f1 = pd.DataFrame(columns = ['Dataset', 'Config',  'SRA', 'DIF-SR', 'Reweighing'])
    count = 0

    for name in datasets:
        for config in algorithms:
            ''' abre os resultados '''
            if os.path.exists('../experimento/' + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv('../experimento/' + name + '_' + config + '_validation.csv', 
                                         sep=';', index_col=0) 
                test = pd.read_csv('../experimento/' + name + '_' + config + '_test.csv', 
                                   sep=';', index_col=0)
                
                ''' Calculate ratio odds '''
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                
                ''' Calculate multicriteria '''
                validation = multicriteria_validation(validation)

                for clf_name in list(validation.index.unique()):
                    
                    values_valid = []
                    values_valid.append(name)
                    values_valid.append(config)
                    
                    values_f1 = []
                    values_f1.append(name)
                    values_f1.append(config)
                    
                    aux_val = validation.loc[clf_name]
                    
                    for preprocessing in ['without', 'dif', 'reweighing']:

                        aux_validation = aux_val.loc[aux_val['preprocessing'].str.contains(preprocessing)].copy()
                        aux_validation = aux_validation.sort_values(by=['clf_name', 'preprocessing'])
                        aux_validation = aux_validation.reset_index()
                        
                        idx = aux_validation[measure].idxmax()
                        values_valid.append(aux_validation.loc[idx, measure])
                        values_f1.append(aux_validation.loc[idx, 'f1_macro'])

                    results_valid.loc[count] = values_valid
                    f1.loc[count] = values_f1
                    count += 1
                
    return results_valid, f1

def report_dif_model(measure):

    results_valid = pd.DataFrame(columns = ['Dataset', 'Config', 'DT', 'KNN', 'MLP'])
    results_test = pd.DataFrame(columns = ['Dataset', 'Config',  'DT', 'KNN', 'MLP'])
    f1 = pd.DataFrame(columns = ['Dataset', 'Config',  'DT', 'KNN', 'MLP'])
    count = 0

    for name in datasets:
        for config in algorithms:
            ''' abre os resultados '''
            if os.path.exists('../experimento/' + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv('../experimento/' + name + '_' + config + '_validation.csv', 
                                         sep=';', index_col=0) 
                test = pd.read_csv('../experimento/' + name + '_' + config + '_test.csv', 
                                   sep=';', index_col=0)
                
                ''' Calculate ratio odds '''
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2
                
                ''' Calculate multicriteria '''
                validation = multicriteria_validation(validation)
                test = multicriteria_validation(test)

                values_test = []
                values_test.append(name)
                values_test.append(config)
                
                values_f1 = []
                values_f1.append(name)
                values_f1.append(config)

                for preprocessing in ['dif_dt', 'dif_knn', 'dif_mlp']:

                    aux_validation = validation.loc[validation['preprocessing'].str.contains(preprocessing)].copy()
                    aux_validation = aux_validation.sort_values(by=['clf_name', 'preprocessing'])
                    aux_validation = aux_validation.reset_index()

                    aux_test = test.loc[test['preprocessing'].str.contains(preprocessing)].copy()
                    aux_test = aux_test.sort_values(by=['clf_name', 'preprocessing'])
                    aux_test = aux_test.reset_index()

                    idx = aux_validation[measure].idxmax()
                    values_test.append(aux_test.loc[idx, measure])
                    values_f1.append(aux_test.loc[idx, 'f1_macro'])
                    
                
                results_test.loc[count] = values_test
                f1.loc[count] = values_f1
                count += 1
                
    return results_test, f1

def cd(results, conjunto):
    df = results[['SRA', 'DIF-SR', 'Reweighing']]
    algorithms_names = df.columns
    results_array = df.values
    ranks_test = np.array([rankdata(-p) for p in results_array])
    average_ranks = np.mean(ranks_test, axis=0)
    print('\n'.join('({}) Rank average: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))  
    # This method computes the critical difference for Nemenyi test
    cd = compute_CD(average_ranks, n=len(df), alpha='0.05', test='nemenyi')
    print('CD = ', cd)
    # This method generates the plot.
    graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=4, textspace=1.25, reverse=False, 
                filename = conjunto + '.pdf')

