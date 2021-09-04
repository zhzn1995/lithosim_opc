python search_hsd.py \
    --data_path ~/Projects/iccad_hotspot/iccad_2012_dct/ \
    --val_set 2-5 \
    --name hsd-mg --dataset iccad19 --fealen 32 \
    --gpus all --batch_size 256 --workers 0 --print_freq 50 \
    --w_lr 0.05 --w_lr_min 0.004 --alpha_lr 0.0012 \
    --input_size 32