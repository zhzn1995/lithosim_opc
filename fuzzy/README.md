## Fuzzy patterns 

Here are some options of the script.

- frac: The moving frac for the input sample.
- input_file: TXT files with coordinates of the polygons such as `1_0_0.txt`.
- opc: `1` for OPC and `0` for lithography simulation.
- gpu_ind: The gpu index you want to use.
- alpha & beta: Hyperparameters in the Sigmoid function.
- step_size: Step size for updating mask image.
- init_bound & init_bias: Initial bound & bias of the initial mask.
- momentum: Momentum hyperparameter for updating the mask image.
- max_step: Max iteration number.

```bash
python fuzzy.py \
        --input_file /path/to/your/inputfile \
        --opc 1 \
        --gpu_ind 2 \
        --alpha 50 \
        --beta 50 \
        --step_size 5e6 \
        --init_bound 1 \
        --init_bias 0.5 \
        --momentum 0.05 \
        --max_step 200 \
        --frac 0.3
```