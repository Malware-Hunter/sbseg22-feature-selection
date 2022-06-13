from sklearn.ensemble import RandomForestClassifier

def correlacao(X, y, k, method, output_file, methods):
    feature_scores = methods[method]['function'](X, y, k)
    new_X = X[list(feature_scores['features'])]
    
    correlation = new_X.corr()
    
    model_RF=RandomForestClassifier()
    model_RF.fit(new_X,y)
    
    feats = {}
    for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
        feats[feature] = importance

    to_drop = set()

    for index in correlation.index:
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.85:
               ft = column if feats[column] <= feats[index] else index
               to_drop.add(ft)
    print("qtd de features removidas:", len(to_drop))

    new_X = new_X.drop(columns = to_drop)
    print("Dataset final criado")
    new_X.to_csv(output_file, index=False)
