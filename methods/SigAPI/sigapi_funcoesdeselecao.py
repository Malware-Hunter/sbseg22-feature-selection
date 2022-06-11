import pandas as pd
import numpy as np
import matplotlib.pyplot
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
import sys
from argparse import ArgumentParser
from methods.utils import get_base_parser, get_dataset, get_X_y

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument('-t', '--threshold', type = float, default = 0.001,
        help = 'Threshold for the minimal range suggestion heuristic. This is the threshold for the difference between the slope of consecutive moving averages of each selection method\'s metrics. Default: 0.001')
    parser.add_argument( '-w', '--window-size', type = int, default = 5,
        help = 'Moving average window size used in the minimal range suggestion heuristic. Default: 5')
    parser.add_argument( '-f', '--initial-n-features', type = int, default = 1,
        help = 'Initial number of features. Default: 1')
    parser.add_argument( '-i', '--increment', type = int, default = 1,
        help = 'Value to increment the initial number of features. Default: 1')
    args = parser.parse_args(argv)
    return args

def get_moving_average(data, window_size=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t=0.001, window_size=5):
    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis=0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

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
    precision = precision_score(y_test, resultado_teste, zero_division = 0)
    recall = recall_score(y_test, resultado_teste, zero_division = 0)
    f1 = f1_score(y_test, resultado_teste, zero_division = 0)

    metricas = [acuracia,precision,recall,f1]
    return metricas

l_mutualInformation = [[0,0,0,0,0]]
l_selectRandom = [[0,0,0,0,0]]
l_selectExtra= [[0,0,0,0,0]]
l_RFERandom = [[0,0,0,0,0]]
l_RFEGradient = [[0,0,0,0,0]]
l_selectKBest= [[0,0,0,0,0]]
if __name__=="__main__":
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    num_features = parsed_args.initial_n_features
    increment = parsed_args.increment
    if(num_features > total_features):
        print(f"ERRO: --initial-n-features ({num_features}) maior que a qtd de features do dataset ({total_features})")
        exit(1)
    while num_features < (total_features + increment):
        k = total_features if num_features > total_features else num_features

        print("qtd de features: ", k)
        mutualinformationGain = calculateMutualInformationGain(X, y, k)
        new_X = X[list(mutualinformationGain['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_mutualInformation = np.append(l_mutualInformation,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        randomForestClassifier = calculateRandomForestClassifier(X, y, k)
        new_X = X[list(randomForestClassifier['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_selectRandom = np.append(l_selectRandom,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        extraTreesClass = calculateExtraTreesClassifier(X, y, k)
        new_X = X[list(extraTreesClass['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_selectExtra = np.append(l_selectExtra,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        RFERandomForestClassifier = calculateRFERandomForestClassifier(X,y, k)
        new_X = X[list(RFERandomForestClassifier['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_RFERandom= np.append(l_RFERandom,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        RFEGradientBoostingClassifier = calculateRFEGradientBoostingClassifier(X,y, k)
        new_X = X[list(RFEGradientBoostingClassifier['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_RFEGradient = np.append(l_RFEGradient,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        selectKBest = calculateSelectKBest(X,y,k)
        new_X = X[list(selectKBest['features'])]
        result_metricas =  calculateMetricas(new_X,y)
        l_selectKBest = np.append(l_selectKBest,[[k,result_metricas[0],result_metricas[1],result_metricas[2],result_metricas[3]]],axis=0)

        num_features += increment

columns = ['Número de Características','Acurácia','Precisão','Recall','F1 Score']

methods = {
    "mutualInformation": pd.DataFrame(l_mutualInformation, columns=columns),
    "selectRandom": pd.DataFrame(l_selectRandom, columns=columns),
    "selectExtra": pd.DataFrame(l_selectExtra, columns=columns),
    "RFERandom": pd.DataFrame(l_RFERandom, columns=columns),
    "RFEGradient": pd.DataFrame(l_RFEGradient, columns=columns),
    "selectKBest": pd.DataFrame(l_selectKBest, columns=columns)
}
for method_name, df in methods.items():
    df.to_csv(f'data-{method_name}-{parsed_args.output_file}.csv', index=False)
    df.drop(columns=['Número de Características']).plot().get_figure().savefig(f'graph-{method_name}-{parsed_args.output_file}.jpg', dpi=300)

lower_bounds = []
try:
    lower_bounds = []
    for method_name, df in methods.items():
        lower_bound = get_minimal_range_suggestion(df, t=parsed_args.threshold, window_size=parsed_args.window_size)
        lower_bounds.append((method_name, lower_bound))
    print(lower_bounds)
    if(len(lower_bounds) == 0):
        print("Não foi possível encontrar o limite inferior do intervalo mínimo.")
        exit(0)
    min_lower_bound = lower_bounds[0]
    for (method_name, lower_bound) in lower_bounds:
        print("lower_bound:", lower_bound)
        if(lower_bound < min_lower_bound[1]):
            min_lower_bound = (method_name, lower_bound)
    if(min_lower_bound[1] == -1):
        print("Não foi possível encontrar o limite inferior do intervalo mínimo.")
    else:
        print("Menor limite inferior encontrado:")  
        print(f'{min_lower_bound[0]}, {min_lower_bound[1]}')
except Exception as e:
    print(str(e))
    print("Não foi possível calcular a sugestão de intervalo mínimo desta vez.")
