python augment_hsd.py --name hsd-mg \
 --val_set 0 --dataset iccad19 --fealen 32 --input_size 32 \
 --gpus all --batch_size 128 --workers 16 --lr 0.025 --print_freq 50 \
 --genotype "Genotype(normal=[[('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], [('skip_connect', 2), ('dil_conv_5x5', 0)], [('skip_connect', 3), ('skip_connect', 2)], [('skip_connect', 2), ('skip_connect', 3)]], normal_concat=range(2, 6), reduce=[[('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], [('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], [('avg_pool_3x3', 0), ('sep_conv_3x3', 2)], [('skip_connect', 0), ('dil_conv_3x3', 2)]], reduce_concat=range(2, 6))"