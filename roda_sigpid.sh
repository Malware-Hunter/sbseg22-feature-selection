CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
CHECK_PKGS=$(pip show numpy scipy pandas scikit-learn mlxtend | grep -i -w "not found")
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install numpy scikit-learn scipy pandas mlxtend"; exit; }

if [[ `ls -1 datasets/*.csv 2>/dev/null | wc -l ` -eq 0 ]]; then
  echo "ERRO: não foi possível encontrar arquivos CSV no diretório \"datasets\"."
  exit 1
fi
for DATASET in datasets/*.csv
do
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    echo "python3 methods/SigPID/sigpid.py -d $DATASET "
    python3 methods/SigPID/sigpid.py -d $DATASET
done