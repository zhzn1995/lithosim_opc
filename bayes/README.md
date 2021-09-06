## NAS-based Hotspot Detection

First prepare your DCT-based ICCAD benchmark with [this code](https://github.com/phdyang007/dlhsd).

Next run the script to search with BO:

```bash
python bayes_search.py \
    --name hsd-mg \
    --val_set 0 \
    --layers 8 \
    --dataset iccad19 \
    --fealen 32 \
    --input_size 32 \
    --gpus all \
    --batch_size 256 \
    --workers 8 \
    --lr 0.05 \
    --print_freq 50

```



- layers: Pre-defined layer number in the block.
- dataset: Dataset name
- fealen: Feature length of the DCT dataset
- Input_size: The shape of the input tensor.
- gpus: The GPU you want to use, `0,1` or `2` or `all` are accepted.
- batch_size: Batch size for training.
- workers: Number of threads for processing the input data.
- lr: Learning rate.
- print_freq: The frequency you want to output your log.