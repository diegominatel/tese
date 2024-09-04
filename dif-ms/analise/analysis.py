import pandas as pd
import numpy as np
import math
import sys
import os
from IPython.display import display
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

mcpm_measures = ['ratio_selection_rate', 'ratio_recall', 'ratio_odds']

def multicriteria(measurement_values):
    pairs = list(combinations(measurement_values, 2))
    area = 0
    for a, b in pairs:
        area += (a*b*math.sin((2*math.pi)/3)/2)    
    return area

def multicriteria_validation(validation):
    matrix = validation[mcpm_measures].to_numpy()
    validation['multicriteria'] = [multicriteria(row) for row in matrix]
    return validation

def general_results(datasets, algorithms):
    measures = ['f1_macro', 'ratio_f1_macro', 'ratio_selection_rate', 'ratio_recall', 'ratio_odds']
    df = pd.DataFrame(columns = ['Dataset', 'Config', 'Selection'] + measures)

    count = 0

    for name in datasets:  
        for config in algorithms:
            if os.path.exists('../experimento/' + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv('../experimento/' + name + '_' + config + '_validation.csv', sep=';', index_col=0)
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                validation = multicriteria_validation(validation)
                
                test = pd.read_csv('../experimento/' + name + '_' + config + '_test.csv', sep=';', index_col=0)
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2
                test = multicriteria_validation(test) 

                idx_DIF1 = validation['DIF_1PL'].idxmin()
                idx_DIF2 = validation['DIF_2PL'].idxmin()
                idx_DIF3 = validation['DIF_3PL'].idxmin()
                idx_MC = validation['multicriteria'].idxmax()
                idx_DP = validation['ratio_selection_rate'].idxmax()

                dif1 = []
                dif2 = []
                dif3 = []
                multi = []
                single = []

                dif1.append(name)
                dif2.append(name)
                dif3.append(name)
                multi.append(name)
                single.append(name)
                
                dif1.append(config)
                dif2.append(config)
                dif3.append(config)
                multi.append(config)
                single.append(config)

                dif1.append('DIF-MS(ML1)')
                dif2.append('DIF-MS(ML2)')
                dif3.append('DIF-MS(ML3)')
                multi.append('MCJ')
                single.append('OCJ')

                for measure in measures:
                    dif1.append(test.loc[idx_DIF1, measure])
                    dif2.append(test.loc[idx_DIF2, measure])
                    dif3.append(test.loc[idx_DIF3, measure])
                    multi.append(test.loc[idx_MC, measure])
                    single.append(test.loc[idx_DP, measure])

                df.loc[count] = dif1 
                count += 1
                df.loc[count] = dif2 
                count += 1
                df.loc[count] = dif3 
                count += 1
                df.loc[count] = multi 
                count += 1
                df.loc[count] = single
                count += 1
                
    return df

def general_statistical(datasets, algorithms, measure, columns):

    df_statistical = pd.DataFrame(columns = ['DIF-MS(ML1)', 'DIF-MS(ML2)', 'DIF-MS(ML3)', 'MCJ', 'OCJ'])

    count = 0

    for name in datasets:    
        for config in algorithms:
            if os.path.exists('../experimento/' + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv('../experimento/' + name + '_' + config + '_validation.csv', sep=';', index_col=0)
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                validation = multicriteria_validation(validation)

                test = pd.read_csv('../experimento/' + name + '_' + config + '_test.csv', sep=';', index_col=0)
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2
                test = multicriteria_validation(test) 

                idx_DIF1 = validation['DIF_1PL'].idxmin()
                idx_DIF2 = validation['DIF_2PL'].idxmin()
                idx_DIF3 = validation['DIF_3PL'].idxmin()
                idx_MC = validation['multicriteria'].idxmax()
                idx_DP = validation['ratio_selection_rate'].idxmax()

                df_statistical.loc[count] = [test.loc[idx_DIF1, measure], test.loc[idx_DIF2, measure], 
                                             test.loc[idx_DIF3, measure], test.loc[idx_MC, measure],
                                             test.loc[idx_DP, measure]]
                count += 1
    return df_statistical

def general_cd(datasets, algorithms, measure, columns, name):
    
    results = general_statistical(datasets, algorithms, measure, columns)
    df = results[columns]
    algorithms_names = df.columns
    results_array = df.values

    ranks_test = np.array([rankdata(-p) for p in results_array])

    average_ranks = np.mean(ranks_test, axis=0)
    print('\n'.join('({}) Média dos ranks: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))
    
    # This method computes the critical difference for Nemenyi test
    cd = compute_CD(average_ranks, n=len(df), alpha='0.05', test='nemenyi')
    print('CD = ', cd)
    # This method generates the plot.
    graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=4, textspace=1.25, reverse=False, 
                filename = '../experimento/' + name + '.pdf')
