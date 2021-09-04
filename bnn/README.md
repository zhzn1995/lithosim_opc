# Setup
### First setup the modified mxnet.

Use ``cmake`` to build the project. Make sure to install all the dependencies described [here](mxnet/docs/install/build_from_source.md#prerequisites). 

Adjust settings in cmake (build-type ``Release`` or ``Debug``, configure CUDA, OpenBLAS or Atlas, OpenCV, OpenMP etc.)  

```shell
$ cd mxnet
$ mkdir build/Release && cd build/Release
$ cmake ../../
$ ccmake . 
$ make -j `nproc`
```

### Build the MXNet Python binding

```shell
$ cd mxnet/python
$ pip install --upgrade pip
$ pip install -e .
```
# Train on ICCAD contest
First run the training script and store the best performed checkpoint.
```shell
cd my
python train.py
```
Then modify ``common/data.py`` and ``train.py`` to finetune the model using dataset with biased labels.
