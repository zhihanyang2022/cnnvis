<img src="https://github.com/zhihanyang2022/pngs/blob/master/cnnvis_banner.png" alt="drawing">

## Welcome

CNNVis is a high-level convolutional neural network (CNN) visualization API built on top of Keras. The intention behind this project aligns with the intention of Keras: "Being able to go from idea to result with the least possible daly is key to doing good research". 

Use CNNVis if you need to visualize the following aspects of a CNN (a `keras.models.Sequential` or `keras.models.Model` instance):
* Kernels ([2 lines of code](#plot-kernels))
* Activations / feature maps of a specific layer to a specific image ([3 lines of code](#plot-activations))
* The 2D pattern that maximally activates a kernel ([1 line of code](#plot-max-activation-image))
* Saliency maps ([1 line of code](#plot-saliency-map))
* Email yangz2@carleton.edu about any additional features that you would like to visualize!

The main resources that tremendously helped the development of this project are:
* Chapter 5: Computer Vision, of book _Deep Learning with Python_ by Francis Chollet
* The official Keras documentation: https://keras.io/

<u>Important Note</u>: This library only supports your Keras if your backend is TensorFlow.

## Getting Started: 30 seconds to CNNVis

First, make sure that all dependencies are installed (`pip install <library-name>` is recommended):
* Numpy
* matplotlib
* prettytable
* Keras

### Initiate a Visualizer instance

To instantiate a `Visualizer` instance for the vgg16 network:
```python
vgg16_model = keras.applications.VGG16(weights='imagenet', include_top=True)
visualizer = Visualizer(model=vgg16_model, input_shape=(1, 224, 224, 3) # (batch_size, height, width, num_channels)
```

### Print summary 

To print the **default summary** of the vgg16 network:
```python
visualizer.model_summary(style='default')
```

To print the **"cnn style" summary** of the vgg16 network:
```python
visualizer.model_summary(style='cnn')
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

### Plot kernels
To plot kernels / filters:
```python
layer_name = 'block1_conv1' # find the layer_name of the layer of interest in the zeroth column heading of "cnn style" model summary
visualizer.get_kernels(layer_name, style='plots') # returns nothing, only plots
```

To obtain kernels / filters as a **tensor** with dimension (kernel_index, height, width, num_channels), pass `'tensors'` as the value to parameter `style` instead:
```python
kernels = visualizer.get_kernels(layer_name, style='tensors') # returns a tensor, plots nothing
```
<img src="https://github.com/zhihanyang2022/pngs/blob/master/kernels.png" alt="drawing" width="500"/>

## Plot saliency maps

To plot saliency maps:

```python
import numpy as np
from matplotlib import pyplot as plt

img_paths = ['fish.jpg', 'bird.jpg', 'elephants.jpg']  # put more img_paths in this list to obtain multiple saliency maps
saliency_maps = visualizer.get_saliency_map(img_paths)  

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
```

<img src="https://github.com/zhihanyang2022/pngs/blob/master/saliency_maps.png" alt="drawing" width="500"/>

### Plot feature maps

To plot feature maps of a specific layer to a specific image (e.g. giraffe):

<img src="https://github.com/zhihanyang2022/pngs/blob/master/giraffe.jpeg" alt="drawing" width="400"/>

```python
import numpy as np
from matplotlib import pyplot as plt

feature_map = visualizer.get_feature_maps(['block5_conv3'], ['giraffe.png'])
plt.matshow(np.mean(feature_map[0, 0], axis=-1))
plt.show()
```
<img src="https://github.com/zhihanyang2022/pngs/blob/master/feature_map_giraffe.png" alt="drawing" width="200"/>

### Plot mean activations

To plot mean activations of multiple layers to multiple images (e.g. ![a cat image](https://github.com/zhihanyang2022/pngs/blob/master/cat.png) and ![a dog image](https://github.com/zhihanyang2022/pngs/blob/master/dog.jpg)):

```python
mean_activation = visualizer.get_mean_activations(['block5_conv2', 'block5_conv3'], [img_path_cat, img_path_dog])
```

```python
plt.plot(mean_activation[0, 0], label='Cat', alpha=0.6)
plt.plot(mean_activation[0, 1], label='Dog', alpha=0.6)
plt.xlabel('Kernel Index')
plt.ylabel('Mean Activation')
plt.title('Mean activations of block5_conv2')
plt.legend()
plt.show()
```

<img src="https://github.com/zhihanyang2022/pngs/blob/master/mean_activations_1.png" alt="drawing" width="400"/>

```python
plt.plot(mean_activation[1, 0], label='Cat', alpha=0.6)
plt.plot(mean_activation[1, 1], label='Dog', alpha=0.6)
plt.xlabel('Kernel Index')
plt.ylabel('Mean Activation')
plt.title('Mean activations of block5_conv3')
plt.legend()
plt.show()
```

<img src="https://github.com/zhihanyang2022/pngs/blob/master/mean_activations_2.png" alt="drawing" width="400"/>


### Plot max activation

To plot max 

