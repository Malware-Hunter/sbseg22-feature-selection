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
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    echo "python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET -o resultado-selecao-$D_NAME"
    OUTPUT=`python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET -o resultado-selecao-$D_NAME`
    if [[ `echo $OUTPUT | grep -o 'Menor limite inferior encontrado:'` = "" ]]; then
        echo -n "Informe o método para a etapa de correlação 
        (mutualInformation, selectRandom, selectExtra, RFERandom, RFEGradient, selectKBest): "
        read METHOD
        echo -n "Informe o limite inferior do intervalo mínimo: "
        NUM_FEATURES
    else
        METHOD=`tail -1 | sed 's/ //g' | cut -d, -f1`
        NUM_FEATURES=`tail -1 | sed 's/ //g' | cut -d, -f2`
    fi
    echo "python3 -m methods.SigAPI.sigapi_correlacao -d $DATASET -k $NUM_FEATURES -m $METHOD -o resultado-correlacao-$D_NAME"
    python3 -m methods.SigAPI.sigapi_correlacao -d $DATASET -k $NUM_FEATURES -m $METHOD -o resultado-correlacao-$D_NAME
done