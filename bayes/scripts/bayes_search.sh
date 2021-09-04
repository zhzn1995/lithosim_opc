python bayes_search.py --name hsd-mg \
    --val_set 0 \
    --layers 8 \
    --dataset iccad19 --fealen 32 --input_size 32 \
    --gpus all --batch_size 256 --workers 8 --lr 0.05 --print_freq 50