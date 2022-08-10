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
    echo "python3 -m metodos.SigAPI.main -d $DATASET -o resultado-selecao-$D_NAME -i $INCREMENT"
    python3 -m metodos.SigAPI.main -d $DATASET -o resultado-selecao-$D_NAME -i $INCREMENT
}

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1
[[ $1 ]] || { echo "Uso: bash $0 DATASET [DATASET...]" && exit 1;}
for DATASET in $*
do
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    TS=$(date +%Y%m%d%H%M%S)
    { time sigapi $DATASET $D_NAME; } 2> time-sigapi-$D_NAME-$TS.txt
done
