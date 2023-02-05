export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH="$PWD"
DIR="logs"
if [ ! -d "$DIR" ]; then
    mkdir logs
fi
# nohup python3 paragon/train.py > logs/train.log 2>&1 &
python3 paragon/train.py