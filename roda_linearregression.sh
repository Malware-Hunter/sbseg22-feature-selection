#!/bin/bash
CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
PKGS=(pandas numpy scikit-learn)
CHECK_PKGS=`pip show ${PKGS[@]} | grep -i -w "not found"`
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install ${PKGS[@]}"; exit; }

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1
for DATASET in datasets/*.csv
do
    echo "python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET"
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    python3 -m methods.LinearRegression.LinearRegression -d $DATASET --output-file linear-regression-$D_NAME
done