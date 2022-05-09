# JOWMDroid
Espaço para a reprodução do trabalho JOWMDroid

_______________________________________________________________________________________________________________
## Implementação do JOWMDroid

### 1 - Seleção de Datasets
Foi selecionado o Drebin-215.

### 2 - Utilizamos o Ganho de Informação (IG)
Foi utilizado o método "mutual_info_regression" da biblioteca sklearn para selecionar descartar as características menos relevantes para os modelos.

### 3 - Modelos para definir pesos de características
Foi utilizado três modelos que contém métodos que distribuem pesos para as características através de seus cálculos.

Os modelos selecionados são:

   * SVM >> método "_coef"

   * Random Forest >> método "feature_importances_"

   * Logistic Regression >> método "_coef"

### 4 - Normalização dos pesos
Após termos definidos os pesos das características através dos três modelos citados acima, notamos que cada um retorna valores diferentes para os pesos, então normalizamos os dados entre 0 e 1 através do cálculo:

Peso_final = (Peso - Peso_min) / (Peso_max - Peso_min)

Onde:

   * Peso     >> é o peso da respectiva característica;

   * Peso_min >> é o menor peso definido;

   * Peso_max >> é o maior peso definido;

### 5 - Média dos pesos
Após normalizar os pesos dos três modelos, tiramos a média deles.

### 6 - Funções de Mapeamento de Pesos
Em seguida, utilizamos 5 funções para mapear os pesos iniciais para pesos finais, sendo elas:

   1° Função de Potência;

   2° Função Exponencial;

   3° Função Logarítmica;

   4° Função Hiperbólica;

   5° Função Curva em S.

### 7 - Algoritmo de Evolução Diferencial (DE)
Nessa etapa, devemos otimizar em conjunto os parâmetros da função de mapeamento de peso e do classificador final através desse algoritmo de evolução diferencial.

(Em andamento)