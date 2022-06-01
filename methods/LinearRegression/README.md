# LinearRegression
Reprodução do experimento do paper [A novel permission-based Android malware detection system using
feature selection based on linear regression](https://link.springer.com/article/10.1007/s00521-021-05875-1).

## Descrição
1. A regressão linear é um método estatístico usado para modelar a relação entre duas ou mais variáveis. No modelo gerado para estimar a variável dependente, chama-se regressão simples se for utilizada uma variável independente simples como entrada, e regressão múltipla se for utilizada mais de uma variável independente. Neste estudo, as permissões do aplicativo correspondem às variáveis independentes, enquanto as variável dependente representa o tipo de aplicativos.
2. A seleção de recursos do sistema de detecção de malware proposto, visa remover recursos desnecessários usando uma abordagem de seleção de recursos baseada em regressão linear. Dessa forma, a dimensão do vetor de recursos é reduzida, o tempo de treinamento é reduzido e o modelo de classificação pode ser usado em sistemas de detecção de malware em tempo real.


## Como instalar

## Dependências 
```
## Instale os pacotes necessários:
pip install notebook numpy pandas scikit-learn
```

## Como rodar

- Após clonar o repositório, inicie o Jupyter Notebook no diretório do método `LinearRegression`:
```
cd feature_selection/methods/LinearRegression/ 
```
