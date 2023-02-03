export PYTHONPATH="$PWD"
nohup python paragon/train.py > logs/train.log 2>&1 &