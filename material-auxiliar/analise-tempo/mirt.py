# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd

''' Load the methods to use R '''
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

''' Load the 'mirt' package in R '''
mirt_package = importr('mirt')
mirt_package.mirtCluster()

class Mirt:
    '''
    Class that runs the mirt package in R
    '''
    
    def __init__(self):
        self.__model = None
        
    def mirt_config(self, itemtype):
        '''
        Returns the parameters used in mirt
        '''
        technical = ro.r('list(NCYCLES = 500, message=FALSE, warn=FALSE)') 
        params = dict() 
        params['technical'] = technical
        params['itemtype'] = itemtype
        params['verbose'] = True
        
        return params
        
    def calculate(self, item_modeling, itemtype):
        '''
        Args:
                item_modeling:  (pd.DataFrame) where rows indicate individuals and columns indicate items, each value pair i,j 
                                represents whether individual i correctly answered item j.
        '''
        item_names = list(item_modeling.columns)
        ''' Convert DataFrame to R Frame '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            item_modeling = ro.conversion.py2rpy(item_modeling)
        ''' Run mirt '''
        self.__model = mirt_package.mirt(item_modeling, 1, **self.mirt_config(itemtype))
        
