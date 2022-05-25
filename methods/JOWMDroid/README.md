# JOWMDroid
Espaço para a reprodução do trabalho "[JOWMDroid: Android malware detection based on feature weighting with joint optimization of weight-mapping and classifier parameters](https://www.sciencedirect.com/science/article/pii/S016740482030359X)".

## Como instalar

```
## 1) Clone o respositório:
git clone https://github.com/Malware-Hunter/feature_selection.git

## 2) Instale as dependências:
pip install numpy scikit-learn scipy pandas
```

## Como rodar

Para rodar o experimento sobre algum dataset (e.g. `data.csv`) execute o seguinte comando:

```
python3 JOWMDroid.py -d data.csv
```

Note: o JOWMDroid assume que o dataset já está pré-processado, conforme consta na seção a seguir.

## Detalhes de uso
```
usage: JOWMDroid.py [-h] -d DATASET [--sep SEPARATOR] [--exclude-hyperparameter] [-m LIST]
                    [-t MI_THRESHOLD] [--train-size TRAIN_SIZE] [-o OUTPUT_FILE] [--cv INT]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed, with the last feature being the class
  --sep SEPARATOR       Dataset feature separator. Default: ","
  --exclude-hyperparameter
                        If set, the ML hyperparameter will be excluded in the Differential Evolution. By default it's included
  -m LIST, --mapping-functions LIST
                        List of mapping functions to use. Default: "power, exponential,logarithmic, hyperbolic, S_curve"
  -t MI_THRESHOLD, --mi-threshold MI_THRESHOLD
                        Threshold to select features with Mutual Information. Default: 0.05. Only features with score greater than or equal to this value will be selected
  --train-size TRAIN_SIZE
                        Proportion of samples to use for train. Default: 0.8
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file name. Default: results.csv
  --cv INT              Number of folds to use in cross validation. Default: 5
  --feature-selection-only
                        If set, the experiment is constrained to the feature selection phase only.
```