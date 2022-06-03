echo "Install Pandas..."
python3 -m pip install pandas
echo "Install scikit-learn..."
python3 -m pip install scikit-learn
echo "Install mlxtend..."
python3 -m pip install mlxtend

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