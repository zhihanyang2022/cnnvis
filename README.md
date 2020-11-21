<img src="https://i.loli.net/2020/11/21/cbltiD4EXY1eQdy.png" alt="drawing">

## Welcome

CNNVis is a high-level convolutional neural network (CNN) visualization API built on top of Keras. The intention behind this project aligns with the intention of Keras: "Being able to go from idea to result with the least possible delay is key to doing good research". 

Use CNNVis if you need to visualize the following aspects of a CNN. Of course, your CNN model needs to be a `keras.models.Sequential` or `keras.models.Model` instance.

* [Kernels](#plot-kernel)
* [Feature maps](#plot-feature-map)
* [Mean activations](#plot-mean-activations)
* [Max activations](#plot-max-activation)
* [Saliency maps](#plot-saliency-map)

Feel free to email me at yangz2@carleton.edu about any additional features that you would like to visualize!

The main resources that helped the development of this project tremendously:
* Chapter 5: Computer Vision, of book _Deep Learning with Python_ by Francis Chollet
* The official Keras documentation: https://keras.io/

<u>Important Note</u>: This library only supports your Keras if your backend is TensorFlow.

## Getting Started: 30 seconds to CNNVis

First, make sure that all dependencies are installed:
* prettytable
* Numpy
* matplotlib
* PIL
* Keras

### First Step: Instantiate a Visualizer instance

To instantiate a `Visualizer` instance for a vgg16 network:

```python
import keras
from cnnvis import Visualizer

vgg16_model = keras.applications.VGG16(weights='imagenet', include_top=True)
visualizer = Visualizer(model=vgg16_model, image_shape=(224, 224, 3), batch_size=1, preprocess_style='vgg16')
```

### Print summary 

To print the **default summary**:
```python
visualizer.summary(style='default')
```

To print the **"cnn style" summary**:
```python
visualizer.summary(style='cnn')
```

```
CNN Style Model Summary
+--------------+--------------+------------+-------------+----------------+---------------------+
|  Layer Name  |  Layer Type  | Kernel Num | Kernel Size | Kernel Padding |     Output Shape    |
+--------------+--------------+------------+-------------+----------------+---------------------+
| block1_conv1 |    Conv2D    |     64     |    (3, 3)   |      same      |  (1, 224, 224, 64)  |
| block1_conv2 |    Conv2D    |     64     |    (3, 3)   |      same      |  (1, 224, 224, 64)  |
| block1_pool  | MaxPooling2D |     /      |    (2, 2)   |       /        |   (1, 112, 112, 3)  |
| block2_conv1 |    Conv2D    |    128     |    (3, 3)   |      same      |  (1, 224, 224, 128) |
| block2_conv2 |    Conv2D    |    128     |    (3, 3)   |      same      |  (1, 224, 224, 128) |
| block2_pool  | MaxPooling2D |     /      |    (2, 2)   |       /        |   (1, 112, 112, 3)  |
| block3_conv1 |    Conv2D    |    256     |    (3, 3)   |      same      |  (1, 224, 224, 256) |
| block3_conv2 |    Conv2D    |    256     |    (3, 3)   |      same      |  (1, 224, 224, 256) |
| block3_conv3 |    Conv2D    |    256     |    (3, 3)   |      same      |  (1, 224, 224, 256) |
| block3_pool  | MaxPooling2D |     /      |    (2, 2)   |       /        |   (1, 112, 112, 3)  |
| block4_conv1 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block4_conv2 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block4_conv3 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block4_pool  | MaxPooling2D |     /      |    (2, 2)   |       /        |   (1, 112, 112, 3)  |
| block5_conv1 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block5_conv2 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block5_conv3 |    Conv2D    |    512     |    (3, 3)   |      same      |  (1, 224, 224, 512) |
| block5_pool  | MaxPooling2D |     /      |    (2, 2)   |       /        |   (1, 112, 112, 3)  |
|   flatten    |   Flatten    |     /      |      /      |       /        |     (1, 150528)     |
|     fc1      |    Dense     |     /      |      /      |       /        | (1, 224, 224, 4096) |
|     fc2      |    Dense     |     /      |      /      |       /        | (1, 224, 224, 4096) |
| predictions  |    Dense     |     /      |      /      |       /        | (1, 224, 224, 1000) |
+--------------+--------------+------------+-------------+----------------+---------------------+
Training set: 1000 Categories of ImageNet
Number of Conv2D layers: 13
Number of MaxPooling2D layers: 5
Number of Dense layers: 3
```

## Plot saliency map

To plot saliency maps:

```python
img_paths = ['fish.jpg', 'bird.jpg', 'elephants.jpg']
saliency_maps = visualizer.get_saliency_map(img_paths)  
```

```python
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.axis('off')
plt.imshow(saliency_maps[0])

fig.add_subplot(1, 3, 2)
plt.axis('off')
plt.imshow(saliency_maps[1])

fig.add_subplot(1, 3, 3)
plt.axis('off')
plt.imshow(saliency_maps[2])

plt.show()
```

<img src="https://i.loli.net/2020/11/21/RQUGPnT7XxNkWa9.png" alt="drawing" width="500"/>

### Plot feature map

To plot the feature map of a specific layer to a specific image (e.g. giraffe):

<img src="https://i.loli.net/2020/11/21/WorY5tJ78OLsBuK.jpg" alt="drawing" width="400"/>

```python
feature_map = visualizer.get_feature_maps(['block5_conv3'], ['giraffe.png'])
```

```python
import numpy as np
from matplotlib import pyplot as plt

plt.matshow(np.mean(feature_map[0, 0], axis=-1))
plt.show()
```
<img src="https://i.loli.net/2020/11/21/afu6hvsNoYwxb9C.png" alt="drawing" width="200"/>

### Plot mean activations

To plot mean activations of multiple layers to multiple images (e.g. a cat image and a dog image):

- https://github.com/zhihanyang2022/pngs/blob/master/cat.png
- https://github.com/zhihanyang2022/pngs/blob/master/dog.jpg

```python
mean_activation = visualizer.get_mean_activations(['block5_conv2', 'block5_conv3'], [img_path_cat, img_path_dog])
```

```python
from matplotlib import pyplot as plt

plt.plot(mean_activation[0, 0], label='Cat', alpha=0.6)
plt.plot(mean_activation[0, 1], label='Dog', alpha=0.6)
plt.xlabel('Kernel Index')
plt.ylabel('Mean Activation')
plt.title('Mean activations of block5_conv2')
plt.legend()
plt.show()
```

<img src="https://i.loli.net/2020/11/21/f729zOmErAFK6gs.png" alt="drawing" width="400"/>

```python
from matplotlib import pyplot as plt

plt.plot(mean_activation[1, 0], label='Cat', alpha=0.6)
plt.plot(mean_activation[1, 1], label='Dog', alpha=0.6)
plt.xlabel('Kernel Index')
plt.ylabel('Mean Activation')
plt.title('Mean activations of block5_conv3')
plt.legend()
plt.show()
```

<img src="https://i.loli.net/2020/11/21/5GpO3B1dc6lg9o8.png" alt="drawing" width="400"/>

### Plot max activation

To plot max activation to specific kernels in a specific layer:

```python
max_activations = visualizer.get_max_activations('block3_conv1', [12, 123], 2)
```

```python
from matplotlib import pyplot as plt

plt.imshow(max_activation[0])
plt.axis('off')
plt.show()
```

<img src="https://i.loli.net/2020/11/21/5GpO3B1dc6lg9o8.png" alt="drawing" width="400"/>

```python
from matplotlib import pyplot as plt

plt.imshow(max_activation[1])
plt.axis('off')
plt.show()
```

<img src="https://i.loli.net/2020/11/21/ZrQHL1oEPSnIDs7.png" alt="drawing" width="400"/>

### Plot kernel

To plot kernels

```python
kernels = visualizer.get_kernels('block2_conv1')
```

```python
from matplotlib import pyplot as plt
import numpy as np

plt.matshow(np.mean(kernels[:, :, :, 1], axis=-1))
plt.show()
```

<img src="https://i.loli.net/2020/11/21/cAZ9dFSK4wn8Plq.png" alt="drawing" width="300"/>



