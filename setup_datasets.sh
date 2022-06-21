#!/bin/bash
# Script para extrair .zip

for DATASET in datasets/*.zip
do
    unzip -n $DATASET -d datasets
done

for DATASET in datasets/*.rar
do
    unrar e -o- $DATASET datasets
done

if [[ `ls -1 datasets/*.csv 2>/dev/null | wc -l ` -eq 0 ]]; then
  echo "ERRO: não foi possível encontrar arquivos CSV no diretório \"datasets\"."
  exit 1
fi
