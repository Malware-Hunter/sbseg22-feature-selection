CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
CHECK_PKGS=$(pip show numpy scipy pandas scikit-learn | grep -i -w "not found")
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install numpy scikit-learn scipy pandas"; exit; }

if [[ `ls -1 datasets/*.csv 2>/dev/null | wc -l ` -eq 0 ]]; then
  echo "ERRO: não foi possível encontrar arquivos CSV no diretório \"datasets\"."
  exit 1
fi

roda_dataset() {
    DATASET=$1
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    echo "python3 methods/JOWMDroid/JOWMDroid.py --feature-selection-only -d $DATASET --output-file jowmdroid-$D_NAME... "
    python3 methods/JOWMDroid/JOWMDroid.py --feature-selection-only -d $DATASET --output-file jowmdroid-$D_NAME
}

for DATASET in datasets/*.csv
do
    roda_dataset $DATASET
done

for DATASET in datasets/*.rar
do
    unrar $DATASET
    DATASET=$(echo $DATASET | sed 's/.rar/.csv/')
    roda_dataset $DATASET
done

