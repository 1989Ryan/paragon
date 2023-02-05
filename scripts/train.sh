export PYTHONPATH="$PWD"
DIR="logs"
if [ ! -d "$DIR" ]; then
    mkdir logs
fi
nohup python paragon/train.py > logs/train.log 2>&1 &