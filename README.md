# CNNVis: Visualizing CNNs

Welcome to CNNVis! CNNVis is a high-level convolutional neural network (CNN) visualization API built on top of Keras. The intention behind this project aligns with the intention of Keras: "Being able to go from idea to result with the least possible daly is key to doing good research". 

Use CNNVis if you need to visualize the following aspects of a CNN (a `keras.models.Sequential` or `keras.models.Model` instance):
* Kernels / filters ([Do this in 2 lines of code](#plot-kernels))
* Activations / feature maps of a specific layer to a specific image ([Do this in 3 lines of code](#plot-activations))
* The 2D pattern that maximally activates a kernel
* Saliency maps
* (Email yangz2@carleton.edu about anything you would like me to add!)

The main resource that tremendously helped the development of this project is:
* The official Keras documentation: https://keras.io/
* Chapter 5: Computer Vision of _Deep Learning with Python_ by Francis Chollet

<u>Important Note</u>: This library only supports a TensorFlow backend for Keras.

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

To obtain kernels / filters as a **tensor** with dimension (index, height, width, num_channels), pass `'tensors'` as the value to parameter `style` instead:
```python
kernels = visualizer.get_kernels(layer_name, style='tensors') # returns a tensor, plots nothing
```
<img src="https://github.com/zhihanyang2022/pngs/blob/master/kernels.png" alt="drawing" width="500"/>

### Plot activations
To plot activations / feature maps of a specific layer to a specific image:
```python
layer_name = 'block1_conv1' # find the layer_name of the layer of interest in the zeroth column heading of "cnn style" model summary
img_path = '/Users/yangzhihan/datasets/cats_and_dogs_dataset/test/cats/1780.jpg' # an example path
visualizer.get_activations(layer_name, img_path, style='plots') # returns nothing, only plots
```
<img src="https://github.com/zhihanyang2022/pngs/blob/master/activations.png" alt="drawing" width="500"/>

To obtain activations / feature maps as a **tensor** with dimension (index, height, width, num_channels), pass `'tensors'` as the value to parameter `style` instead:
```python
visualizer.get_activations(layer_name, img_path, style='tensors') # returns a tensor, plots nothing
```

### Plot max activation image
### Plot saliency map
