#sudo pip install numpy scikit-learn scipy pandas

for DATASET in datasets/*.csv
do
    echo -n "python3 methods/JOWMDroid/JOWMDroid.py -d $DATASET ... "
    python3 methods/JOWMDroid/JOWMDroid.py --feature-selection-only -d $DATASET
    echo "done."
done
