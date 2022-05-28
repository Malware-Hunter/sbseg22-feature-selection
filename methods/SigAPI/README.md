# SigAPI

Neste espaço foi realizada a reprodução do artigo [Significant API Calls in Android Malware Detection](https://ksiresearch.org/seke/seke20paper/paper143.pdf).
Este artigo utiliza técnicas de seleção de características e a correlação baseada na eliminação destas features.

# Descrição
1 - Para selecionar as features são utilizadas 5 funções de seleção. Dentre estas cinco, a que possui o menor intervalo, neste caso, apresenta maior eficiência é a escolhida.

2 - Depois de encontrar a função mais eficiente para determinado conjunto de dados, é feita a correlação no intervalo indicado, para haver uma maior redução destas características.

## Como instalar

```
## Clone o respositório:
git clone https://github.com/Malware-Hunter/feature_selection.git
## Instale as seguintes dependências:
pip install seaborn pandas numpy scikit-learn
```

## Como rodar

Primeiro iremos rodar as funções de selecao `sigapi_funcoesdeselecao.py`
Para rodar esta parte do experimento sobre algum dataset (e.g. `data.csv`) execute o seguinte comando:

```
python3 sigapi_funcoesdeselecao.py -d data.csv
``` 
         
Com isso, vão ser obtidos gráficos e dataframes sobre cada uma das 6 técnicas de seleção utilizadas.
Ao analisar esses dados, é possivel obter a técnica mais eficiente e seu intervalo de redução.

Após, é necessário rodar o código `sigapi_correlação.py` , fazendo o seguinte:

```
python3 sigapi_correlacao.py -d data.csv
``` 
            
Com isso, vai ser possível encontrar a redução de características que foi realizada e o resultado será um dataset com estas características.
  
