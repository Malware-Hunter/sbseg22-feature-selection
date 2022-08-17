# SBSeg 22 - Feature Selection

## Uma Analise de Métodos de Seleção de Características Aplicados a Detecção de Malwares Android

Repositório de códigos, conjuntos de dados (*datasets*) e arquivos usados na elaboração do *paper*.

Os diretórios deste repositório estão organizados da seguinte forma:

###### papers
Artigos de origem dos métodos de seleção implementados.

###### methods
Códigos dos métodos.

###### datasets
Datasets construídos para o estudo.


## Exemplos de uso:

- Executar o método **SigAPI** para todos os *datasets* no diretorio **datasets**
  ```sh
  $ bash roda_sigapi.sh
  ```

- Executar o método **SigAPI** para todos os *datasets* no diretorio **datasets/balanceados**
  ```sh
  $ bash roda_sigapi_bl.sh
  ```

- Executar o modelo **Random Forest (RF)** para o *dataset* **resultado_sigpid_drebin_215_all.csv**
  ```sh
  $ python3 roda_ml.py -d resultado_sigpid_drebin_215_all.csv -c rf
  ```

- Executar o modelo **SVM** para o *dataset* **resultado_sigpid_drebin_215_all.csv**
  ```sh
  $ python3 roda_ml.py -d resultado_sigpid_drebin_215_all.csv -c rf
  ```
