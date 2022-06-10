#!/bin/bash
CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
PKGS=(pandas numpy scikit-learn)
CHECK_PKGS=`pip show ${PKGS[@]} | grep -i -w "not found"`
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install ${PKGS[@]}"; exit; }

set_increment(){
    TOTAL_FEATURES=$1
    [[ $TOTAL_FEATURES -lt 50 ]] && INCREMENT=1 && return
    [[ $TOTAL_FEATURES -lt 1000 ]] && INCREMENT=5 && return
    INCREMENT=10
}

sigapi(){
    DATASET=$1
    D_NAME=$2
    set_increment `head -1 $DATASET | awk -F, '{print NF-1}'`
    echo "python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET -o resultado-selecao-$D_NAME -i $INCREMENT"
    OUTPUT=`python3 -m methods.SigAPI.sigapi_funcoesdeselecao -d $DATASET -o resultado-selecao-$D_NAME -i $INCREMENT`
    if [[ `echo $OUTPUT | grep -o 'Menor limite inferior encontrado:'` = "" ]]; then
        echo -n "Informe o método para a etapa de correlação 
        (mutualInformation, selectRandom, selectExtra, RFERandom, RFEGradient, selectKBest): "
        read METHOD
        echo -n "Informe o limite inferior do intervalo mínimo: "
        NUM_FEATURES
    else
        METHOD=`echo $OUTPUT | tail -1 | sed 's/ //g' | cut -d, -f1`
        NUM_FEATURES=`echo $OUTPUT | tail -1 | sed 's/ //g' | cut -d, -f2`
    fi
    echo "python3 -m methods.SigAPI.sigapi_correlacao -d $DATASET -k $NUM_FEATURES -m $METHOD -o resultado-correlacao-$D_NAME"
    python3 -m methods.SigAPI.sigapi_correlacao -d $DATASET -k $NUM_FEATURES -m $METHOD -o resultado-correlacao-$D_NAME
}

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1
[[ $1 ]] || { echo "Uso: bash $0 DATASET [DATASET...]" && exit 1;}
for DATASET in $*
do
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    TS=$(date +%Y%m%d%H%M%S)
    { time sigapi $DATASET; } 2> time-sigapi-$D_NAME-$TS.txt
done
