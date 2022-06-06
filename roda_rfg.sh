echo "Preparando ambiente virtual ..."

VENV="venv-rfg"
PYTHON="${VENV}/bin/python3"
PIP="${VENV}/bin/pip3"
BASE_DIR="methods/RFG"
CANT_INSTALL_VENV_MESSAGE="ERRO: não foi possível criar o ambiente virtual (venv).\nO python3.8 e o python3.8-venv estão instalados? Se não, use o comando: sudo apt install python3.8 python3.8-venv"

if [[ `which javac java | wc -l` -lt 2 ]]; then
  echo "ERRO: instale o javac e java primeiro (e.g.: sudo apt install -y openjdk-11-jdk) não foram encontrados.">&2
  exit 1
fi

if ! [[ -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  [[ $? == 1 || ! -d "$VENV" ]] && echo -e "$CANT_INSTALL_VENV_MESSAGE">&2 && exit 1
fi
[[ ! -f $PIP ]] && echo "ERRO: ${PIP} não encontrado">&2 && exit 1

$PIP install numpy==1.22.3 wheel
$PIP install -r $BASE_DIR/requirements.txt

echo "Ambiente virtual preparado em \"${VENV}\""

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1

MAX_N_FEATURES=100
for DATASET in datasets/*.csv
do
    TOTAL_N_FEATURES=`head -1 "$DATASET" | awk -F, '{print NF}'`
    [[ $TOTAL_N_FEATURES -gt $MAX_N_FEATURES ]] && N_FEATURES=$MAX_N_FEATURES || N_FEATURES=`expr $TOTAL_N_FEATURES - 1`
    echo "$PYTHON $BASE_DIR/rfg.py -d $DATASET -f $N_FEATURES --feature-selection-only"
    $PYTHON -m methods.RFG.rfg -d $DATASET -f $N_FEATURES --feature-selection-only
    echo "Done"
done