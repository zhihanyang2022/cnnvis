# TODO: add some diagrams and describe more methods

# CNNVis: Visualizing CNNs

Welcome to CNNVis! CNNVis is a high-level convolutional neural network (CNN) visualization API built on top of Keras. The intention behind this project aligns with the intention of Keras: "Being able to go from idea to result with the least possible daly is key to doing good research". 

Use CNNVis if you need to visualize the following aspects of a CNN (a `keras.models.Sequential` or `keras.models.Model` instance):
* Kernels / filters
* Activations / feature maps of a specific layer to a specific image
* The 2D pattern that maximally activates a kernel
* Saliency maps
* (Email yangz2@carleton.edu about anything you would like to visualize!)

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

The `Visualizer` class is the only class defined in this library. 

To instantiate a `Visualizer` for visualizing aspects of the vgg16 network (or any other `keras.models.Sequential` and `keras.models.Model` instances:
```
vgg16_model = keras.applications.VGG16(weights='imagenet', include_top=True)
visualizer = Visualizer(model=vgg16_model, model_input_shape=(1, 224, 224, 3) # (batch_size, height, width, num_channels)
```

To **print the default summary** of the vgg16 network:
```
visualizer.model_summary(style='default')
```

To **print the "cnn style" summary** (including the number of kernels, the size of kernels and the style of padding for each layer)  of the vgg16 network:
```
visualizer.model_summary(style='cnn')
```

To **plot kernels / filters**:
```
layer_name = 'block1_conv1' # find the layer names in the zeroth column heading of "cnn style" model summary
visualizer.get_kernels(layer_name, style='plots')
```

<img src="https://github.com/zhihanyang2022/pngs/blob/master/activations.png" alt="drawing" width="750"/>

To **obtain kernels / filters as a tensor** with dimension (index, height, width, num_channels), pass `'tensors'` as the value to parameter `style` instead:
```
kernels = visualizer.get_kernels(layer_name, style='tensors')
```

To **plot activations / feature maps of a specific layer to a specific image**:
```
layer_name = 'block1_conv1' # find the layer names in the zeroth column heading of "cnn style" model summary
img_path = '/Users/yangzhihan/datasets/cats_and_dogs_dataset/test/cats/1780.jpg'
visualizer.get_activations(layer_name, img_path, style='plots')
```

To **obtain activations / feature maps as a tensor** with dimension (index, height, width, num_channels), pass `'tensors'` as the value to parameter `style` instead:
```
visualizer.get_activations(layer_name, img_path, style='tensors')
```
