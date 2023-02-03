export PYTHONPATH="$PWD"
nohup python paragon/train.py > logs/train_raven.log 2>&1 &