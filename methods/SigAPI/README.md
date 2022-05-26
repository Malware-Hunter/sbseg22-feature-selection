# SigAPI

Neste espaço foi realizada a reprodução do artigo [Significant API Calls in Android Malware Detection](https://ksiresearch.org/seke/seke20paper/paper143.pdf).
Este artigo utiliza técnicas de seleção de características e a correlação baseada na eliminação destas features.

# Descrição
1 - Para selecionar as features são utilizadas 5 funções de seleção. Dentre estas cinco, a que possui o menor intervalo, neste caso, apresenta maior eficiência é a escolhida.

2 - Depois de encontrar a função mais eficiente para determinado conjunto de dados, é feita a correlação no intervalo indicado, para haver uma maior redução destas características.

## Como instalar
```
## Instale as seguintes dependências:
pip install seaborn pandas numpy scikit-learn
```

## Como rodar

Ao baixar o código `sigapi_funçõesdeseleção.py` é possível mudar o dataset com que se deseja trabalhar na linha 113 onde há: 
         
         dataset = pd.read_csv('Drebin_215_CPI.csv')  
         
Com isso, vão ser obtidos gráficos e dataframes sobre cada uma das 6 técnicas de seleção utilizadas.
Ao analisar esses dados, é possivel obter a técnica mais eficiente e seu intervalo de redução.

Após, é necessário baixar o código `sigapi_correlação.py` , onde na linha 113 é possível mudar o dataset, nas linhas 117 - 119 você altera o intervalo de redução para o encontrado, exemplo 

          num_features = 18 
    
          for k in range(num_features,26)
            
Com isso, vai ser possível encontrar a redução que foi realizada e as features que foram selecionadas como as mais significantes.
  
