python augment_hsd.py --name hsd-mg \
    --val_set 0 \
    --layers 15 \
    --dataset iccad19 --fealen 32 --input_size 32 \
    --gpus all --batch_size 128 --workers 8 --lr 0.05 --print_freq 50 \
    --genotype "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 0)], [('sep_conv_3x3', 2), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], [('sep_conv_5x5', 4), ('sep_conv_5x5', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], [('sep_conv_5x5', 2), ('dil_conv_5x5', 1)], [('dil_conv_3x3', 1), ('sep_conv_3x3', 3)], [('sep_conv_3x3', 4), ('sep_conv_5x5', 2)]], reduce_concat=range(2, 6))"
    # --data_path ~/Projects/iccad_hotspot/iccad_2012_dct/ \
