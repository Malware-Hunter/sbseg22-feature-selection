# -*- coding: utf-8 -*-
"""Funções SigAPI.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DWpzEOgHkYdHgeG14mgmRDEnoe8DiCBU
"""
"""
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest  , chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import csv
import timeit
"""
"""# **Correlação**"""

import matplotlib.pyplot as plt 
if __name__=="__main__":
    dataset = pd.read_csv('drebin_sigapi.csv')
    X = dataset.drop(columns = ['class']) #variaveis (features)
    y = dataset['class'] #classification eh a classificacao de benignos e malwares
    total_features = dataset.shape[1] - 1 #CLASS
    num_features = 18 
    
    for k in range(num_features,26):
       
       
        print(">>> NÚMERO DE FEATURES ",k, "<<<")
        

        print(">>> RFE USING RANDOM FOREST CLASSIFIER <<<")
        RFERandomForestClassifier = calculateRFERandomForestClassifier(X,y, k)
        new_X = X[list(RFERandomForestClassifier['features'])]
        print(RFERandomForestClassifier['features'])
       
        correlation = new_X.corr()
        plot = sn.heatmap(correlation, annot = True, fmt=".2f", linewidths=.9)
        plot.figure.set_size_inches(12, 8)
        plt.show()

        model_RF=RandomForestClassifier()
        model_RF.fit(new_X,y)
        RF_weights= model_RF.feature_importances_
        print(RF_weights)
        feats = {} # a dict to hold feature_name: feature_importance
        
        for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
            feats[feature] = importance #add the name/value pair

        to_drop = set()

        for index in correlation.index:
            for column in correlation.columns:
                if index != column and correlation.loc[index, column] > 0.85:
                    ft = column if feats[column] <= feats[index] else index
                    to_drop.add(ft)
        print("PARA REMOVER >>", to_drop)

        new_X = new_X.drop(columns = to_drop)
        print(new_X)
