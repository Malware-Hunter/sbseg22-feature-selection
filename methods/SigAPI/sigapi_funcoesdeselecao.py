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
    parser.add_argument('-c', '--class_column', type = str, default="class", metavar = 'CLASS_COLUMN', 
        help = 'Name of the class column. Default: "class"')
    parser.add_argument('-n', '--n-samples', type=int,
        help = 'Use a subset of n samples from the dataset. RFG uses the whole dataset by default.')
    return parser.parse_args(sys.argv[1:])

"""# **Função Incremento** """

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

l_mutualInformation = [[0,0,0,0,0]]
l_selectRandom = [[0,0,0,0,0]]
l_selectExtra= [[0,0,0,0,0]]
l_RFERandom = [[0,0,0,0,0]]
l_RFEGradient = [[0,0,0,0,0]]
l_selectKBest= [[0,0,0,0,0]]
if __name__=="__main__":
    args = parse_args()
    dataset = pd.read_csv(args.dataset, sep=args.sep)
    n_samples = args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            print(f"Error: expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
            sys.exit(1)
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)
    X = dataset.drop(columns = ['class'])
    if(args.class_column not in dataset.columns):
        print(f'ERRO: dataset não possui uma coluna chamada "{args.class_column}"')
        exit(1)
    y = dataset[args.class_column]
    total_features = dataset.shape[1] - 1
    num_features = 1
    increment = 1
    while num_features < (total_features + increment):
        k = total_features if num_features > total_features else num_features
       
        print(">>> NÚMERO DE FEATURES ",k, "<<<")
        print(">>> MUTUAL INFORMATION GAIN <<<")
        mutualinformationGain = calculateMutualInformationGain(X, y, k)
        new_X = X[list(mutualinformationGain['features'])]

        result_metricas =  calculateMetricas(new_X,y)
        l_mutualInformation = np.append(l_mutualInformation,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_mutualInformation)

        a_mutualInformation = open('Metricas_MutualGain.csv', 'w', newline='', encoding='utf-8')
        w_mutualInformation = csv.writer(a_mutualInformation)
        w_mutualInformation.writerow([l_mutualInformation])
        arquivo_mutualInformation = open('Metricas_MutualGain.csv')
        
        print(">>> SELECTFROMMODEL USING RANDOM FOREST CLASSIFIER <<<")
        randomForestClassifier = calculateRandomForestClassifier(X, y, k)
        new_X = X[list(randomForestClassifier['features'])]

        result_metricas =  calculateMetricas(new_X,y)
        l_selectRandom = np.append(l_selectRandom,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_selectRandom)
        a_selectRandom = open('Metricas_SelectRFC.csv', 'w', newline='', encoding='utf-8')
        w_selectRandom = csv.writer(a_selectRandom)
        w_selectRandom.writerow([l_selectRandom])
        arquivo_selectRandom = open('Metricas_SelectRFC.csv')

        print(">>> SELECTFROMMODEL USING EXTRA TREES CLASSIFIER <<<")
        extraTreesClass = calculateExtraTreesClassifier(X, y, k)
        new_X = X[list(extraTreesClass['features'])]

        result_metricas =  calculateMetricas(new_X,y)
        l_selectExtra = np.append(l_selectExtra,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_selectExtra)
        a_selectExtra = open('Metricas_SelectETC.csv', 'w', newline='', encoding='utf-8')
        w_selectExtra = csv.writer(a_selectExtra)
        w_selectExtra.writerow([l_selectExtra])
        arquivo_selectExtra = open('Metricas_SelectETC.csv')
        
        print(">>> RFE USING RANDOM FOREST CLASSIFIER <<<")
        RFERandomForestClassifier = calculateRFERandomForestClassifier(X,y, k)
        new_X = X[list(RFERandomForestClassifier['features'])]
        
        result_metricas =  calculateMetricas(new_X,y)
        l_RFERandom= np.append(l_RFERandom,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_RFERandom)
        a_RFERandom = open('Metricas_RFERandom.csv', 'w', newline='', encoding='utf-8')
        w_RFERandom = csv.writer(a_RFERandom)
        w_RFERandom.writerow([l_RFERandom])
        arquivo_RFERandom = open('Metricas_RFERandom.csv')

        print(">>> RFE USING GRADIENT BOOSTING CLASSIFIER <<<")
        RFEGradientBoostingClassifier = calculateRFEGradientBoostingClassifier(X,y, k)
        new_X = X[list(RFEGradientBoostingClassifier['features'])]

        result_metricas =  calculateMetricas(new_X,y)
        l_RFEGradient = np.append(l_RFEGradient,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_RFEGradient)
        a_RFEGradient = open('Metricas_RFEGradient.csv', 'w', newline='', encoding='utf-8')
        w_RFEGradient= csv.writer(a_RFEGradient)
        w_RFEGradient.writerow([l_RFEGradient])
        arquivo_RFEGradient = open('Metricas_RFEGradient.csv')
       
        print(">>> SELECT K BEST <<<")
        selectKBest = calculateSelectKBest(X,y,k)
        new_X = X[list(selectKBest['features'])]

        result_metricas =  calculateMetricas(new_X,y)
        l_selectKBest = np.append(l_selectKBest,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)
        print(l_selectKBest)

        a_selectKBest = open('Metricas_SelectKBest.csv', 'w', newline='', encoding='utf-8')
        w_selectKBest = csv.writer(a_selectKBest)
        w_selectKBest.writerow([l_selectKBest])
        arquivo_selectKBest = open('Metricas_SelectKBest.csv')

        num_features += increment

df_mutualInformation= pd.DataFrame(l_mutualInformation,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])
df_selectRandom= pd.DataFrame(l_selectRandom,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])
df_selectExtra= pd.DataFrame(l_selectExtra,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])
df_RFERandom= pd.DataFrame(l_RFERandom,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])
df_RFEGradient= pd.DataFrame(l_RFEGradient,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])
df_selectKBest= pd.DataFrame(l_selectKBest,columns=['Número de Características','Acurácia','Precisão','Recall','F1 Score'])

df_mutualInformation.to_csv("data1.csv", index = False)
df_selectRandom.to_csv("data2.csv", index = False)
df_selectExtra.to_csv("data3.csv", index = False)
df_RFERandom.to_csv("data4.csv", index = False)
df_RFEGradient.to_csv("data5.csv", index = False)
df_selectKBest.to_csv("data6.csv", index = False)