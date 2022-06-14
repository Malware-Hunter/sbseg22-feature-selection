from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from logitboost import LogitBoost
import numpy as np
import pandas as pd
import wittgenstein as lw

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.dataset import create_instances_from_matrices
from weka.filters import Filter

from argparse import ArgumentParser
import sys
from methods.utils import get_base_parser, get_dataset, get_X_y

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument(
        '-i', '--increment', 
        help = 'Increment. Default: 20',
        type = int, 
        default = 20)
    parser.add_argument(
        '-f',
        metavar = 'LIST',
        help = 'List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"',
        type = str, 
        default = "")
    parser.add_argument(
        '-k', '--n-folds',
        help = 'Number of folds to use in k-fold cross validation. Default: 10.',
        type = int, 
        default = 10)
    parser.add_argument(
        '-t', '--prediction-threshold', 
        metavar = 'THRESHOLD',
        help = 'Prediction threshold for Weka classifiers. Default: 0.6',
        type = float, 
        default = 0.6)
    parser.add_argument('--feature-selection-only', action='store_true',
        help="If set, the experiment is constrained to the feature selection phase only. The program always returns the best K features, where K is the maximum value in the features list.")
    args = parser.parse_args(argv)
    return args

"""# Execução do experimento

### Observação

Os seguintes algoritmos utilizados no paper possuem implementação na ferramenta Weka, mas não no Scikit-learn: Sequential Minimal Optimization (SMO), AdaBoostM1 e Random Committee. Portanto, vamos usar a biblioteca `python-weka-wrapper3` para eles. No caso do SMO, deve haver um classificador para cada kernel usado no paper, a saber, kernel polinomial, kernel normalizado, kernel PUK e kernel RBF.
"""

def preprocess_instances_to_nominal(instances):
    numeric_to_nominal = Filter("weka.filters.unsupervised.attribute.NumericToNominal")
    numeric_to_nominal.inputformat(instances)
    nominal_instances = numeric_to_nominal.filter(instances)
    return nominal_instances

class WekaClassifier:
    def __init__(self, classifier, preprocess_instances, prediction_threshold=0.6):
        self.classifier = classifier
        self.threshold = prediction_threshold
        self.preprocess_instances = preprocess_instances
    
    def fit(self, X, y):
        train_data = create_instances_from_matrices(np.array(X), np.array(y))
        train_data = self.preprocess_instances(train_data)
        train_data.class_is_last()
        self.classifier.build_classifier(train_data)
    
    def predict(self, X, y):
        test_data = create_instances_from_matrices(np.array(X), np.array(y))
        test_data = self.preprocess_instances(test_data)
        test_data.class_is_last()
        y_pred = [self.classifier.classify_instance(instance) for instance in test_data]
        if(len(set(y_pred)) > 2):
            y_pred = [0 if prob > self.threshold else 1 for prob in y_pred]
        return y_pred

def get_best_features_sorted(selector, columns):
    indices = selector.get_support(indices=True)
    best_scores = selector.scores_[indices]
    best_features = pd.Series([[best_scores[i], indices[i], columns[i]] for i in range(0, len(indices))]).sort_values(ascending=False)
    best_features = best_features.reset_index()
    best_features['feature'] = [line[2] for line in best_features[0]]
    best_features['score'] = [line[0] for line in best_features[0]]
    best_features['feature_index'] = [line[1] for line in best_features[0]]
    best_features = best_features.drop(columns=[0, 'index'])
    return best_features

def run_experiment(X, y, classifiers, is_feature_selection_only = False,
                   score_functions=[chi2, f_classif], 
                   n_folds=10,
                   k_increment=20,
                   k_list=[]):
    """
    Esta função implementa um experimento de classificação binária usando validação cruzada e seleção de características. 
    Os "classifiers" devem implementar as funções "fit" e "predict", como as funções do Scikit-learn.
    Se o parâmetro "k_list" for uma lista não vazia, então ele será usado como a lista das quantidades de características a serem selecionadas. 
    """
    results = []
    best_features = []
    if(len(k_list) > 0):
        k_values = k_list
    else:
        k_values = range(1, X.shape[1], k_increment)
    for k in k_values:
        if(k > X.shape[1]):
            print(f"Warning: skipping K = {k}, since it's greater than the number of features available ({X.shape[1]})")
            continue
        print("K =", k)
        for score_function in score_functions:
            if(k == max(k_values)): 
                selector = SelectKBest(score_func=score_function, k=k).fit(X, y)
                X_selected = X.iloc[:, selector.get_support(indices=True)].copy()
                feature_scores_sorted = pd.DataFrame(list(zip(X_selected.columns.values.tolist(), selector.scores_)), columns= ['features','score']).sort_values(by = ['score'], ascending=False)
                X_selected_sorted = X_selected.loc[:, list(feature_scores_sorted['features'])]
                X_selected_sorted['class'] = y

                best_features.append({
                    'score_function' : score_function.__name__, 
                    'selected_dataset' : X_selected_sorted,
                    'k': k 
                })
                if(X_selected.shape[1] == 1):
                    print("AVISO: 0 features selecionadas")
            if(is_feature_selection_only):
                continue
            X_selected = SelectKBest(score_func=score_function, k=k).fit_transform(X, y)
            kf = KFold(n_splits=n_folds, random_state=256, shuffle=True)
            fold = 0
            for train_index, test_index in kf.split(X_selected):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                for classifier_name, classifier in classifiers.items():
                    classifier.fit(X_train, y_train)
                    if(isinstance(classifier, WekaClassifier)):
                        y_pred = classifier.predict(X_test, y_test)
                    else:
                        y_pred = classifier.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    results.append({'n_fold': fold,
                                    'k': k,
                                    'score_function':score_function.__name__,
                                    'algorithm': classifier_name,
                                    'accuracy': report['accuracy'],
                                    'precision': report['macro avg']['precision'], 
                                    'recall': report['macro avg']['recall'],
                                    'f-measure': report['macro avg']['f1-score']
                                })
                fold += 1
            
    return pd.DataFrame(results), best_features

def main():    
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    k_list = [int(value) for value in parsed_args.f.split(",")] if parsed_args.f != "" else []

    jvm.start()
    classifiers = {
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(metric='euclidean'),
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'SimpleLogistic': LogitBoost(),
        'JRIP': lw.RIPPER(),
        'SMO-PolyKernel' : WekaClassifier(
            Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.PolyKernel']), 
            preprocess_instances_to_nominal, 
            parsed_args.prediction_threshold
            ),
        'SMO-NormalizedPolyKernel': WekaClassifier(
            Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.NormalizedPolyKernel']),
            preprocess_instances_to_nominal,
            parsed_args.prediction_threshold),
        'SMO-Puk': WekaClassifier(
            Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.Puk']),
            preprocess_instances_to_nominal,
            parsed_args.prediction_threshold),
        'SMO-RBFKernel': WekaClassifier(
            Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.RBFKernel']),
            preprocess_instances_to_nominal,
            parsed_args.prediction_threshold),
        'AdaBoostM1': WekaClassifier(
            Classifier("weka.classifiers.meta.AdaBoostM1"),
            preprocess_instances_to_nominal,
            parsed_args.prediction_threshold),
        'RandomCommittee':  WekaClassifier(
            Classifier("weka.classifiers.meta.RandomCommittee"),
            preprocess_instances_to_nominal,
            parsed_args.prediction_threshold)
    }

    results, best_features = run_experiment(
        X, y, 
        classifiers, 
        n_folds = parsed_args.n_folds, 
        k_increment = parsed_args.increment, 
        k_list=k_list, 
        is_feature_selection_only=parsed_args.feature_selection_only
    )

    if(not parsed_args.feature_selection_only):
        results.to_csv(parsed_args.output_file, index=False)
    for best_feature in best_features:
        k = best_feature['k']
        score_function = best_feature['score_function']
        file_name = f"top_{k}_features_with_{score_function}_{parsed_args.output_file}.csv"
        best_feature['selected_dataset'].to_csv(file_name, index=False)
    print("done")

    jvm.stop()
    exit(0)

if __name__ == '__main__':
    main()