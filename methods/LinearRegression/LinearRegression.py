import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import time

#FOLDERS
def KFolders():
    kf = KFold(n_splits=10, shuffle=False) # set division - in 10 folds
    kf.get_n_splits(X) # returns the number of split iterations in the cross validator
    #print(kf)
    return (kf)

#FILTRAGEM DAS FEATURES
def FilterFeatures(ft_names, coef):
  i=0
  n1 = len(coef)
  features_list_to_delete = []

  for i in range(0, n1):
     if coef[i] < 0.1 and coef[i] > -0.1:
       features_list_to_delete.append(ft_names[i])
  return (features_list_to_delete)

# APLICAÇÃO DA REGRESSÃO LINEAR
def LinearR():
    for train_index, test_index in KFolders().split(X):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)                          
        coef_in = model.coef_

        ft_to_delete = FilterFeatures(features_names, coef_in)

        fold_ft_num.append(len(ft_to_delete))
        fold_ft_to_delete.append(ft_to_delete)

        for ft in ft_to_delete:
            index = list(features_names).index(ft)
            #print(index)
            fold_count[index]-=1
        #fold_count
        #print('fold_ft_num', fold_ft_num)
        #print('fold_ft_to_delete', fold_ft_to_delete)
    return(fold_ft_num, fold_ft_to_delete)

# VALOR MÁXIMO DE FEATURES PARA EXCLUSÃO
def MaxValue():
    max_value = None
    index = None

    for idx, num in enumerate(fold_ft_num):
        if max_value is None or num > max_value:
            max_value = num
            index = idx
    #print('value', max_value, 'index', index)
    #print(fold_ft_to_delete[index])
    return(index)

# NOVO DATASET
def NewDataset():
    new_X = X.drop(columns=fold_ft_to_delete[MaxValue()])
    #print(new_X)
    df2 = pd.DataFrame(new_X)
    return(df2)

# LEITURA DO DATASET
dataframe = pd.read_csv('***CAMINHO DO DATASET A SER UTILIZADO****')
print("**********************LEITURA DO DATASET**********************\n", dataframe)
X = dataframe.iloc[:,:-1] # train
y = dataframe.iloc[:,-1] # test

# VETORES AUXILIARES
features_names = np.array(X.columns.values.tolist())
fold_count = [10] * len(features_names)
fold_ft_num = []
fold_ft_to_delete = []

# CHAMADA DAS FUNÇÕES
KFolders()
inicio = time.time()
LinearR()
fim = time.time()
MaxValue()
print("\n\n**********************DATASET COM AS FEATURES SELECIONADAS**********************\n")
print(NewDataset())
print("\n**********************\nTEMPO DE EXECUÇÃO: ", fim-inicio, "\n")


