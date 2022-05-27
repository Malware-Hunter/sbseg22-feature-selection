CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
#sudo pip install numpy scikit-learn scipy pandas

for DATASET in datasets/*.csv
do
    echo "python3 methods/JOWMDroid/JOWMDroid.py --feature-selection-only -d $DATASET ... "
    python3 methods/JOWMDroid/JOWMDroid.py --feature-selection-only -d $DATASET
done
