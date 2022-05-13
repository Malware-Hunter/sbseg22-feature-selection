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

import argparse
import sys
from multiprocessing import Pool

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', 
        help = 'Dataset (csv file). It should be already preprocessed, comma separated, with the last feature being the class.', 
        type = str, 
        required = True)
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
    parser.add_argument(
        '-o', '--output-file', 
        metavar = 'OUTPUT_FILE',
        help = 'Output file name. Default: results.csv',
        type = str, 
        default = 'results.csv')

    args = parser.parse_args(argv)
    return args

"""# Execução do experimento

### Observação

Os seguintes algoritmos utilizados no paper possuem implementação na ferramenta Weka, mas não no Scikit-learn: Sequential Minimal Optimization (SMO), AdaBoostM1 e Random Committee. Portanto, vamos usar a biblioteca `python-weka-wrapper3` para eles. No caso do SMO, deve haver um classificador para cada kernel usado no paper, a saber, kernel polinomial, kernel normalizado, kernel PUK e kernel RBF.
"""

def convert_numeric_to_nominal(instances):
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

def train(classifier, X, y):
    classifier.fit(X, y)
    return classifier

def test(classifier, X, y):
    if(isinstance(classifier, WekaClassifier)):
        y_pred = classifier.predict(X, y)
    else:
        y_pred = classifier.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return {'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'], 
            'recall': report['macro avg']['recall'],
            'f-measure': report['macro avg']['f1-score']
            }

def run_experiment(X, y, classifiers, 
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
    if(len(k_list) > 0):
        k_values = k_list
    else:
        k_values = range(10, X.shape[1], k_increment)
    for k in k_values:
        if(k > X.shape[1]):
            continue
        print("K =", k)
        for score_function in score_functions:
            X_selected = SelectKBest(score_func=score_function, k=k).fit_transform(X, y)
            kf = KFold(n_splits=n_folds, random_state=256, shuffle=True)
            
            indices = [(train_index, test_index) for train_index, test_index in kf.split(X_selected)]
            fitted_classifiers = {'parallel': {}, 'sequential': {}}
            for classifier_name, classifier in classifiers['parallel'].items():
                with Pool() as pool:
                    fitted_classifiers['parallel'][classifier_name] = pool.starmap(train, [(classifier, X_selected[train_index], y[train_index]) for train_index, _ in indices])
            
            for classifier_name, classifier in classifiers['sequential'].items():
                fitted_classifiers['sequential'][classifier_name] = [train(classifier, X_selected[train_index], y[train_index]) for train_index, _ in indices]
            
            print("begin test parallel")
            for classifier_name, fitteds in fitted_classifiers['parallel'].items():
                for fitted in fitteds:
                    with Pool() as pool:
                        test_results = pool.starmap(test, [(fitted, X_selected[test_index], y[test_index]) for _, test_index in indices])
                    for fold in range(len(test_results)):
                        results.append({**test_results[fold], 'k': k, 'n_fold': fold, 'score_function': score_function.__name__, 'algorithm': classifier_name})
            
            print("begin test sequential")
            for _, test_index in indices:
                X_test, y_test = X_selected[test_index], y[test_index]
                for classifier_name, fitteds in fitted_classifiers['sequential'].items():
                    fold = 0
                    for fitted in fitteds:
                        result = test(fitted, X_test, y_test)
                        results.append({**result, 'k': k, 'n_fold': fold, 'score_function': score_function.__name__, 'algorithm': classifier_name})
                        fold += 1
    return pd.DataFrame(results)

def main():
    args = parse_args(sys.argv[1:])

    dataset = pd.read_csv(args.dataset)
    X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
    k_list = [int(value) for value in args.f.split(",")] if args.f != "" else []

    jvm.start()
    classifiers = {
        'parallel': {
            'NaiveBayes': GaussianNB(),
            'KNN': KNeighborsClassifier(metric='euclidean'),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(),
            'DecisionTree': DecisionTreeClassifier(),
            'SimpleLogistic': LogitBoost(),
            'JRIP': lw.RIPPER()
        },
        'sequential': {
            'SMO-PolyKernel' : WekaClassifier(
                Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.PolyKernel']), 
                convert_numeric_to_nominal, 
                args.prediction_threshold),
            'SMO-NormalizedPolyKernel': WekaClassifier(
                Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.NormalizedPolyKernel']),
                convert_numeric_to_nominal,
                args.prediction_threshold),
            'SMO-Puk': WekaClassifier(
                Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.Puk']),
                convert_numeric_to_nominal,
                args.prediction_threshold),
            'SMO-RBFKernel': WekaClassifier(
                Classifier("weka.classifiers.functions.SMO", options=['-K', 'weka.classifiers.functions.supportVector.RBFKernel']),
                convert_numeric_to_nominal,
                args.prediction_threshold),
            'AdaBoostM1': WekaClassifier(
                Classifier("weka.classifiers.meta.AdaBoostM1"),
                convert_numeric_to_nominal,
                args.prediction_threshold),
            'RandomCommittee':  WekaClassifier(
                Classifier("weka.classifiers.meta.RandomCommittee"),
                convert_numeric_to_nominal,
                args.prediction_threshold)
        }
    }

    results = run_experiment(X, y, classifiers, n_folds = args.n_folds, k_increment = args.increment, k_list=k_list)
    results.to_csv(args.output_file)
    print("done")

    jvm.stop()
    exit(0)

if __name__ == '__main__':
    main()