# LinearRegression
Reprodução do experimento do paper [A novel permission-based Android malware detection system using
feature selection based on linear regression](https://link.springer.com/article/10.1007/s00521-021-05875-1).


## Como instalar
```
## Instale os pacotes necessários:
pip install notebook numpy pandas scikit-learn
```

## Como rodar

- Após clonar o repositório, inicie o Jupyter Notebook no diretório do método `LinearRegression`:
```
cd feature_selection/methods/LinearRegression/ && jupyter notebook
```

- Uma página deve ter sido aberta no seu navegador no endereço `localhost:8888/tree`. Nesta página, você deve encontrar o arquivo `LinearRegression.ipynb`. Abra-o clicando nele.

- No notebook que foi aberto defina o dataset a ser utilizado:
```
df = pd.read_csv("<insira o caminho do dataset aqui>/<nome do dataset aqui>.csv")
```

- Execute todas as células de código na ordem em que aparecem.
