import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import csv
import argparse
import sys
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-d', '--dataset', type = str, required = True,
        help = 'Dataset (csv file). It should be already preprocessed, with the last feature being the class')
    parser.add_argument( '--sep', metavar = 'SEPARATOR', type = str, default = ',',
        help = 'Dataset feature separator. Default: ","')
    parser.add_argument('-c', '--class-column', type = str, default="class", metavar = 'CLASS_COLUMN',
        help = 'Name of the class column. Default: "class"')
    parser.add_argument('-n', '--n-samples', type=int,
        help = 'Use a subset of n samples from the dataset. RFG uses the whole dataset by default.')
    parser.add_argument('-k', '--num_features', type = int ,required = True, 
        help = 'Number of features')
    parser.add_argument('-m', '--method', type = str , required = True,
        help = f'One of the following feature selection methods to use: {", ".join(metodos)}')
   
    return parser.parse_args(sys.argv[1:])

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

metodos = { 
    "MutualInformationGain": calculateMutualInformationGain, 
    "RandomForestClassifier": calculateRandomForestClassifier,
    "ExtraTreesClassifier ": calculateExtraTreesClassifier, 
    "RFERandomForestClassifier": calculateRFERandomForestClassifier,
    "RFEGradientBoostingClassifier": calculateRFEGradientBoostingClassifier,
    "SelectKBest": calculateSelectKBest 
}

if __name__=="__main__":
    args = parse_args()
    dataset = pd.read_csv(args.dataset, sep=args.sep)
    n_samples = args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            print(f"Error: expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
            sys.exit(1)
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)

    if(args.class_column not in dataset.columns):
        print(f'ERRO: dataset não possui uma coluna chamada "{args.class_column}"')
        exit(1)
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    total_features = dataset.shape[1] - 1
    k = args.num_features    

    print(">>> MÉTODO MAIS EFICIENTE <<<")
    metodo_eficiente = metodos[args.method](X, y, k)
    new_X = X[list(metodo_eficiente['features'])]
    print(metodo_eficiente['features'])
          
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
    new_X.to_csv("DatasetReduzido.csv", index = False)
