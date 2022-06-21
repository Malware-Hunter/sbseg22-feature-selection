#!/bin/bash
CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
PKGS=(pandas numpy scikit-learn)
CHECK_PKGS=`pip show ${PKGS[@]} | grep -i -w "not found"`
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install ${PKGS[@]}"; exit; }

for DATASET in balanced_datasets/*.csv
do
    echo "python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET"
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    TS=$(date +%Y%m%d%H%M%S)
    { time python3 -m methods.LinearRegression.LinearRegression -d $DATASET --output-file linear-regression-$D_NAME; } 2> time-LR-$D_NAME-$TS.txt
done
