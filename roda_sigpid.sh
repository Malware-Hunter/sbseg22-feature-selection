#!/bin/bash
CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
PKGS=(pandas scikit-learn mlxtend matplotlib)
CHECK_PKGS=`pip show ${PKGS[@]} | grep -i -w "not found"`
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install ${PKGS[@]}"; exit; }

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1
for DATASET in datasets/*.csv
do
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    echo "python3 -m metodos.SigPID.sigpid -d $DATASET -o resultado_sigpid_$D_NAME"
    TS=$(date +%Y%m%d%H%M%S)
    { time python3 -m metodos.SigPID.sigpid -d $DATASET -o resultado_sigpid_$D_NAME; } 2> time-sigpid-$D_NAME-$TS.txt
done
