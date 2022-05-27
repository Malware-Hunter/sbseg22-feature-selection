[ $1 ] && [ -f $1 ] || { echo "Uso: $0 <dataset.csv>"; exit; }

#sudo pip install numpy scikit-learn scipy pandas

python3 methods/JOWMDroid/JOWMDroid.py -d $1

