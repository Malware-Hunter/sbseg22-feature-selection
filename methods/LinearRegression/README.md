# LinearRegression
Reprodução do experimento do paper [A novel permission-based Android malware detection system using
feature selection based on linear regression](https://link.springer.com/article/10.1007/s00521-021-05875-1).

## Descrição
1. A regressão linear é um método estatístico usado para modelar a relação entre duas ou mais variáveis. No modelo gerado para estimar a variável dependente, chama-se regressão simples se for utilizada uma variável independente simples como entrada, e regressão múltipla se for utilizada mais de uma variável independente. Neste estudo, as permissões do aplicativo correspondem às variáveis independentes, enquanto as variável dependente representa o tipo de aplicativos.
2. A seleção de recursos do sistema de detecção de malware proposto, visa remover recursos desnecessários usando uma abordagem de seleção de recursos baseada em regressão linear. Dessa forma, a dimensão do vetor de recursos é reduzida, o tempo de treinamento é reduzido e o modelo de classificação pode ser usado em sistemas de detecção de malware em tempo real.

## Dependências 
O `LinearRegression.py` foi desenvolvido e testado no sistema operacional Ubuntu 22.04 LTS, com a versão da linguagen Python 3.10.4.

## Como instalar
```
## 1) Clone o respositório:
git clone https://github.com/Malware-Hunter/feature_selection.git
```


## Como rodar

Para rodar o experimento sobre algum dataset, basta executar execute o seguinte comando:

```
python3 LinearRegression.py -d dataset.csv
```
Ao final será gerado um arquivo ```results.csv```
