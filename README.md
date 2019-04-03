# CNNVis: Visualizing CNNs
Welcome to CNNVis! CNNVis is a high-level convolutional neural network (CNN) visualization API built on top of Keras. The intention behind this project aligns with the intention of Keras: "Being able to go from idea to result with the least possible daly is key to doing good research". 

Use CNNVis if you need to visualize the following aspects of a CNN:
* Kernels / filters
* Activations of a specific layer / kernel to specific images
* The 2D pattern that maximally activates a kernel
* Saliency maps
* (Email yangz2@carleton.edu about anything you would like to visualize!)

The main resource that tremendously helped the development of this project is:
* The official Keras documentation: https://keras.io/
* Chapter 5: Computer Vision of _Deep Learning with Python_ by Francis Chollet

This library only supports a TensorFlow backend for Keras.

## Getting Started: 30 seconds to CNNVis

First, import Keras:
```
import keras
```

The `Visualizer` class is the only class defined in this library. 

To instantiate a `Visualizer` for visualizing aspects of the vgg16 network:
```
vgg16_model = keras.applications.VGG16(weights='imagenet', include_top=True)
visualizer = Visualizer(model=vgg16_model, model_input_shape=(1, 224, 224, 3) # (batch_size, height, width, num_channels)
```

To print the default summary of the vgg16 network:
```
visualizer.model_summary(style='default')
```

To print 

To plot kernels using matplotlib:
```
layer_name = 'block1_conv1'
visualizer.get_kernels(layer_name, style='plots')
```

To obtain kernels in tensors of dimension (index, height, width, num_channels):
```
kernels = visualizer.get_kernels(layer_name, style='tensors')
```




## Dependencies of CNNVis
* Numpy
* matplotlib
* prettytable
* Keras
