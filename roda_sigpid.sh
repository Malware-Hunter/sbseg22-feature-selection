echo "Install Pandas..."
python3 -m pip install pandas
echo "Install scikit-learn..."
python3 -m pip install scikit-learn
echo "Install mlxtend..."
python3 -m pip install mlxtend

bash setup_datasets.sh
[[ $? != 0 ]] && exit 1
for DATASET in datasets/*.csv
do
    D_NAME=$(echo $DATASET | cut -d"/" -f2)
    echo "python3 methods/SigPID/sigpid.py -d $DATASET "
    python3 methods/SigPID/sigpid.py -d $DATASET
done