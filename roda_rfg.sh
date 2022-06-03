echo "Preparando ambiente virtual ..."

VENV="venv-rfg"
PYTHON="${VENV}/bin/python3"
PIP="${VENV}/bin/pip3"
BASE_DIR="methods/RFG"
CANT_INSTALL_VENV_MESSAGE="ERRO: não foi possível criar o ambiente virtual (venv).\nO python3.8 e o python3.8-venv estão instalados? Se não, use o comando: sudo apt install python3.8 python3.8-venv"

if ! [[ -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  [[ $? == 1 || ! -d "$VENV" ]] && echo -e "$CANT_INSTALL_VENV_MESSAGE">&2 && exit 1
fi
[[ ! -f $PIP ]] && echo "ERRO: ${PIP} não encontrado">&2 && exit 1

$PIP install numpy==1.22.3 wheel
$PIP install -r $BASE_DIR/requirements.txt

echo "Ambiente virtual preparado em \"${VENV}\""
if [[ `ls -1 datasets/*.csv 2>/dev/null | wc -l ` -eq 0 ]]; then
  echo "ERRO: não foi possível encontrar arquivos CSV no diretório \"datasets\"."
  exit 1
fi
for DATASET in datasets/*.csv
do
    DATASET_SHAPE="`wc -l < "$DATASET"`,`head -1 "$DATASET" | awk -F, '{print NF}'`"
    echo -n "qtd de features para o dataset $DATASET (linhas,colunas: $DATASET_SHAPE): "
    read FEATURES_LIST
    echo "$PYTHON $BASE_DIR/rfg.py -d $DATASET -f $FEATURES_LIST --feature-selection-only"
    $PYTHON -m methods.RFG.rfg -d $DATASET -f $FEATURES_LIST --feature-selection-only
    echo "Done"
done