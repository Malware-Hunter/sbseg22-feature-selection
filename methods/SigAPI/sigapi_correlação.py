# -*- coding: utf-8 -*-
"""Funções SigAPI.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DWpzEOgHkYdHgeG14mgmRDEnoe8DiCBU
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


# **Funções**

def calculateMutualInformationGain(features, target, k):
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {"features": feature_names, "score": mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df[:k]

def calculateRandomForestClassifier(features, target,k):
    feature_names= np.array(X.columns.values.tolist())
    test = RandomForestClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df

def calculateExtraTreesClassifier(features, target, k):
    feature_names= np.array(X.columns.values.tolist())
    test = ExtraTreesClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df



def calculateRFERandomForestClassifier(features, target, k):
    feature_names= np.array(X.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df

def calculateRFEGradientBoostingClassifier(features, target,k):
    feature_names= np.array(X.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df


def calculateSelectKBest(features, target,k):
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features,target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score']).sort_values(by = ['score'], ascending=False)
    return df[:k]


def calculateMetricas(new_X,y):
    new_X_train,new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,random_state = 0)

    teste = RandomForestClassifier()
    teste.fit(new_X_train, y_train)
    resultado_teste = teste.predict(new_X_test)

    acuracia = accuracy_score(y_test, resultado_teste)
    precision = precision_score(y_test, resultado_teste)
    recall = recall_score(y_test, resultado_teste)
    f1 = f1_score(y_test, resultado_teste)
   
    metricas = [acuracia,precision,recall,f1]
    return metricas



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
