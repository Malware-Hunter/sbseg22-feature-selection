# RFG

Implementação do experimento do paper [_Automated Malware Detection in Mobile App Stores Based on Robust Feature Generation_](https://doi.org/10.3390/electronics9030435) (RFG). O experimento é composto pelas seguintes etapas:

1. Feature selection por meio do Chi-quadrado e o ANOVA;
    - A quantidade `k` de características varia de forma incremental. Por padrão, incia-se com `k = 10` e segue-se incrementando em 20 até a quantidade total de features do dataset. Você pode definir o valor do incremento ou suprir uma lista com os valores de `k` a serem utilizados.
2. Treino e teste dos seguintes modelos através de validação cruzada _K-fold_: Naive Bayes,
KNN, Random Forest, J48, Sequential Minimal Optimization (SMO), Logistic Regression, AdaBoost decision-stump, Random Committee, JRip e Simple Logistics.


Observação sobre o notebook `RFG.ipynb` e o programa `rfg.py`. Originalmente, o experimento estava sendo desenvolvido via Jupyter Notebook no notebook `RFG.ipynb`. Mas devido à necessidade de se executar o experimento num servidor remoto, suprindo argumentos do experimento (e.g.: dataset, número de folds e etc) por linha de comando, o código do notebook foi convertido para o programa `rfg.py`. O notebook `RFG.ipynb` só está aqui se quisermos implementar alguma melhoria ou se quisermos alterar alguma etapa do experimento, pois é mais prático de se fazer isso no ambiente do Jupyter Notebook. O programador responsável, portanto, deve manter o programa `rfg.py` atualizado de acordo com o notebook `RFG.ipynb`.

# Como rodar

1. Instale as dependências:

```
pip3 install -r requirements.txt
```

2. Para rodar o experimento do RFG sobre algum dataset, use o script `rfg.py`, como no exemplo:

```
python3 rfg.py -d Drebin215.csv
```

Há algumas opções disponíveis, que você pode conferir na seção a seguir.

## Datalhes de uso do `rfg.py`
```
usage: rfg.py [-h] -d DATASET [-i INCREMENT] [-f LIST] [-k N_FOLDS] [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed, comma separated, with the last feature being the class.
  -i INCREMENT, --increment INCREMENT
                        Increment. Default: 20
  -f LIST               List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"
  -k N_FOLDS, --n-folds N_FOLDS
                        Number of folds to use in k-fold cross validation. Default: 10.
  -t THRESHOLD, --prediction-threshold THRESHOLD
                        Prediction threshold for Weka classifiers. Default: 0.6
```