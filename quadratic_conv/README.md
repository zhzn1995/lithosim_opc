## Quadratic convolution

Use Quadratic-VGG-16 in your code:

```python
import tensorflow as tf
from vgg import vgg_16
x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[None])
    is_training = tf.placeholder(tf.bool, shape=[])
logits, endpoints = vgg_16(x, 1000, is_training)
```



Use Quadratic-GoogLeNet in your code:

```python
import tensorflow as tf
from inception import inception_v1
x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[None])
    is_training = tf.placeholder(tf.bool, shape=[])
logits, endpoints = inception_v1(x, 1000, is_training)
```

