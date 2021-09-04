python search_hsd.py \
    --name hsd-mg --dataset iccad19 --val_set 0 --fealen 32 \
    --gpus all --batch_size 128 --workers 0 --print_freq 50 \
    --w_lr 0.025 --w_lr_min 0.004 --alpha_lr 0.0012 \
    --input_size 32