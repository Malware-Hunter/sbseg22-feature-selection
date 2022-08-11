# datasets

## Drebin_215

[Drebin_215 (ORIGINAL)](https://figshare.com/articles/dataset/Android_malware_dataset_for_machine_learning_2/5854653)

## Androcrawl

[Androcrawl (ORIGINAL)](https://github.com/phretor/ransom.mobi/blob/gh-pages/f/filter.7z)

## MD46K

Passo-a-passo da construção do dataset MD46K:

- Etapa 1:  **seleção** dos 46K aplicativos, datados a partir de 2018, do [AndroZoo](https://androzoo.uni.lu/).

- Etapa 2: **download** dos APKs dos 46K aplicativos. O próprio [AndroZoo disponibiliza uma API para o download dos APKs](https://androzoo.uni.lu/api_doc) dos aplicativos.

- Etapa 3: **rotulação** das amostras, utilizando a API do serviço online do [VirusTotal](https://www.virustotal.com), que utiliza mais de 60 scanners para analisar e rotular um aplicativo entre benigno ou malicioso.

- Etapa 4: **extração das características** estáticas dos APKs utilizando a ferramenta [AndroGuard](https://github.com/androguard/androguard).

- Etapa 5: **análise das características** utilizando a documentação oficial da Google, outros datasets existentes e, também, outras ferramentas e serviços de extração, como Malscan, Koodous e SandDroid. 

A sexta etapa foi onde o dataset foi criado de fato, a partir dos dados gerados e validados nas etapas anteriores. O dataset é representado por uma matriz onde 
as linhas são as amostras e as colunas suas características e metadados. A ocorrência de uma determinada característica é representada pelo valor 1 (e.g., o 
aplicativo requisita a permissão para enviar SMS), e a não ocorrência como 0.
Por fim, a sétima etapa do processo foi a sanitização do dataset, onde foi necessário procurar e corrigir problemas e ruídos nos dados, como registros duplicados, 
valores incoerentes, formato inadequado dos dados e valores faltantes.
Todas as etapas são de suma importância para a construção de um dataset com uma grande variedade de dados atuais e que não inclua vieses em seus dados.
