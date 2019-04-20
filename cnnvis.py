"""
cnnvis.py
Zhihan Yang, April 20 2019, Carleton College
"""

import sys
sys.stderr = open('/dev/null', 'w')

from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import keras
import utility as util


class Visualizer:

    def __init__(self, model=None, image_size=(224, 224, 3), batch_size=1):
        """
        Initialize a Visualizer.

        :param model:
            if model is None, then the vgg16 model is used;
            if model is a `keras.models.Sequential` or `keras.models.Model` instance, then the passed-in model is used
        :param image_size: image size including the number of channels, e.g. (224, 224, 3), (150, 150, 3)
        :param batch_size: batch size
        :return None
        """
        if model is None:
            self.model = keras.applications.VGG16(weights='imagenet', include_top=True)
        else:
            if isinstance(model, keras.models.Sequential) or isinstance(model, keras.models.Model):
                self.model = model
            else:
                raise ValueError('The model passed to the `model` parameter is neither a `keras.models.Sequential instance` nor a `keras.models.Model instance`.')

        self.input_shape = (batch_size, ) + image_size  # concatenate batch_size and image_size, obtain shape of tensor
        self.preprocess_style = 'vgg16'

    def summary(self, style='cnn'):
        """
        Print the model summary of `self.model`.

        :param style:
            if style = 'default', then print the default `self.model.summary()`
            if style = 'cnn', then print out the cnn style summary
        :return None
        """
        if style == 'default':
            print(self.model.summary())

        elif style == 'cnn':
            print('CNN Style Model Summary')

            summary = PrettyTable()
            summary.field_names = ["Layer Name",
                                   "Layer Type",
                                   "Kernel Num",
                                   "Kernel Size",
                                   "Kernel Padding",
                                   "Output Shape"]

            num_Conv2D = 0
            num_MaxPooling2D = 0
            num_Dense = 0
            input_shape = self.input_shape
            for layer in self.model.layers:

                if isinstance(layer, keras.layers.Conv2D):
                    num_Conv2D += 1
                    output_shape = layer.compute_output_shape(input_shape)

                    summary.add_row([layer.name,
                                     layer.__class__.__name__,
                                     layer.filters,
                                     layer.kernel_size,
                                     layer.padding,
                                     output_shape])

                    input_shape = output_shape
                elif isinstance(layer, keras.layers.MaxPooling2D):
                    num_MaxPooling2D += 1
                    output_shape = layer.compute_output_shape(input_shape)

                    summary.add_row([layer.name,
                                     layer.__class__.__name__,
                                     '/',
                                     layer.pool_size,
                                     '/',
                                     output_shape])

                    input_shape = output_shape
                elif isinstance(layer, keras.layers.Flatten):
                    output_shape = layer.compute_output_shape(input_shape)

                    summary.add_row([layer.name,
                                     layer.__class__.__name__,
                                     '/',
                                     '/',
                                     '/',
                                     output_shape])

                    input_shape = output_shape
                elif isinstance(layer, keras.layers.Dense):
                    num_Dense += 1
                    output_shape = layer.compute_output_shape(input_shape)

                    summary.add_row([layer.name,
                                     layer.__class__.__name__,
                                     '/',
                                     '/',
                                     '/',
                                     output_shape])

                    input_shape = output_shape
                elif isinstance(layer, keras.layers.Dropout):
                    output_shape = layer.compute_output_shape(input_shape)

                    summary.add_row([layer.name,
                                     layer.__class__.__name__,
                                     '/',
                                     '/',
                                     '/',
                                     output_shape])

                    input_shape = output_shape

            print(summary.get_string())
            print('Number of Conv2D layers: {}'.format(num_Conv2D))
            print('Number of MaxPooling2D layers: {}'.format(num_MaxPooling2D))
            print('Number of Dense layers: {}'.format(num_Dense))

        else:
            raise ValueError('{} is not a valid value for keyword argument "style".'.format(style))

    def get_outputs_of_layers(self, layer_names):
        """
        Return a list of symbolic outputs of specified convolution layers.
        Helper method of `self.get_model_with_output_layers`.

        :param layer_names: a list of layer names
        :return: a list of symbolic outputs of selected convolution layers
        """
        # convert one-item `layer_names` into a list of that one item
        layer_names = util.enforce_list(layer_names)
        # append symbolic output tensors of convolution layers whose names are in `layer_names`
        outputs = []
        for layer_name in layer_names:
            layer = self.model.get_layer(layer_name)  # captures 'layer not found' errors

            if isinstance(layer, keras.layers.Conv2D):
                outputs.append(layer.output)
            else:
                raise ValueError('Layer with name "{}" is not a 2D convolution layer.'.format(layer_name))

        return outputs

    def get_model_with_output_layers(self, layer_names):
        """
        Return a keras.models.Model instance that inputs images and outputs feature maps of specified convolution
        layers.

        :param layer_names: a list of layer names
        :return: a model that outputs feature maps of specified convolution layers when inputted images
        """
        return keras.models.Model(inputs=self.model.input,
                                  outputs=self.get_outputs_of_layers(layer_names))

    # The following section of code is for visualizing kernels.

    def get_kernels(self, layer_name):
        """
        Return the convolution kernels / weights of a layer.

        :param layer_name: the name of the layer whose convolution kernels will be returned
        :return: an numpy array of convolution kernels of the specified layer
        """
        kernels = self.model.get_layer(layer_name).get_weights()[0]
        return kernels

    # The following section of code is for visualizing feature maps.

    def get_feature_maps(self, layer_names, img_paths):
        """
        Return the feature maps outputted by specified layers during prediction of specified images.

        :param layer_names: a list of the name of the layer whose output you want to capture
        :param img_paths: a list of paths to the jpg or png file
        :return: refer to the description of param 'output_style'
        """
        util.enforce_list(layer_names, img_paths)

        model_with_output_layers = self.get_model_with_output_layers(layer_names)
        img_tensors = util.from_img_paths(img_paths, target_size=self.input_shape[1:-1], preprocess=self.preprocess_style)
        feature_maps = model_with_output_layers.predict(img_tensors)

        return util.to_indexable(util.enforce_list(feature_maps))

    # The following section of code is for visualizing mean activations.

    def get_mean_activations(self, layer_names, img_paths):
        """
        Return a list of lists (one for each image) of lists (one for each layer) of mean activations. Each mean
        activation is a scalar representing the degree of activation of a kernel to an image or a feature map.

        :param layer_names: a list of layer names
        :param img_paths: a list of paths of images
        """
        feature_maps = self.get_feature_maps(layer_names, img_paths)

        mean_activations = np.zeros(feature_maps.shape, dtype=np.ndarray)
        for layer, image in np.ndindex(feature_maps.shape):
            mean_activations[layer, image] = np.mean(feature_maps[layer, image], axis=(0, 1))

        return mean_activations

    # The following section of code is for visualizing max activations.

    def loop_kernel_index_for_get_max_activations(func):
        """Help `get_max_activations` loop through kernel indices one at a time."""
        def wrapper(*args):
            outputs = []
            xs = args[2] if isinstance(args[2], list) else [args[2]]
            for x in xs:
                outputs.append(func(args[0], args[1], x, args[3]))
            return outputs
        return wrapper

    @loop_kernel_index_for_get_max_activations
    def get_max_activations(self, layer_name, kernel_index, stds):
        """
        Return the max activation image to a kernel.

        :param layer_name: the layer in which the kernel is located
        :param kernel_index: the index of a kernel
        :param stds: the number of standard deviations in which pixel values are kept
        """
        symbolic_feature_maps = self.model.get_layer(layer_name).output

        # define symbolic gradient descent
        mean_activation = keras.backend.mean(symbolic_feature_maps[:, :, :, kernel_index])
        grad = keras.backend.gradients(mean_activation, self.model.input)[0]
        grad /= (keras.backend.sqrt((keras.backend.mean(keras.backend.square(grad)))) + 1e-5)
        iterate = keras.backend.function([self.model.input], [mean_activation, grad])

        # execute gradient descent
        random_image = np.random.random(self.input_shape) * 20 + 128
        learning_rate = 1
        for i in range(20):
            loss_value, grad_value = iterate([random_image])
            random_image += grad_value * learning_rate

        # ignore the tensor axis because we are only dealing with one image
        return self.to_rgb(random_image[0], stds)

    @staticmethod
    def to_rgb(array, stds):
        """Return an rgb image array suitable for plotting with `matplotlib.pyplot.imshow()`."""
        center = 0.5
        lower_bound, upper_bound = 0, 1
        return np.clip((array - np.mean(array)) / np.std(array) * (0.5 / stds) + center, lower_bound, upper_bound)

    # The following section of code is for visualizing saliency maps.

    def loop_get_saliency_map(func):
        """Help `get_saliency_map` loop through image paths one at a time."""
        def wrapper(*args):
            outputs = []
            img_paths = args[1]
            img_paths = util.enforce_list(img_paths)
            for img_path in img_paths:
                outputs.append(func(args[0], img_path))
            return outputs[0] if (len(outputs) == 1) else outputs
        return wrapper

    @loop_get_saliency_map
    def get_saliency_map(self, img_path, color_map=cm.plasma, output='overlay', intensity=0.6):
        """
        Return the saliency map of an image.

        :param img_path: the path of an image
        :param color_map: the colormap to use for generating saliency heatmap
        :param output:
            if output == 'heatmap', then a saliency heatmap (numpy array) will be returned
            if output == 'overlay' (default), then a saliency map (heatmap + image) (numpy array) will be returned
        :param intensity: the transparency of the heatmap overlaid on the image, relevant when output == 'overlay'
        """
        img_tensor = util.from_img_paths(img_path, target_size=self.input_shape[1:-1], preprocess='vgg16')

        preds = self.model.predict(img_tensor)
        class_index = np.argmax(preds)

        class_output = self.model.output[:, class_index]
        last_conv_layer = self.last_conv_layer

        grads = keras.backend.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))
        iterate = keras.backend.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap_array = np.mean(conv_layer_output_value, axis=-1)
        heatmap_array = np.maximum(heatmap_array, 0)
        heatmap_array /= np.max(heatmap_array)

        if output == 'map':
            return heatmap

        elif output == 'overlay':
            heatmap_img = Image.fromarray(color_map(heatmap_array, bytes=True))
            resized_heatmap_img = heatmap_img.resize((224, 224))
            resized_heatmap_array = np.array(resized_heatmap_img)

            x_array = util.from_img_paths(img_path, target_size=self.input_shape[1:-1], preprocess='normalize')[0]
            x_image = Image.fromarray((x_array * 255).astype('uint8'), 'RGB')
            resized_heatmap_img = Image.fromarray(resized_heatmap_array[:, :, 0:3])

            x_image = x_image.convert("RGBA")
            resized_heatmap_img = resized_heatmap_img.convert("RGBA")

            saliency_map_img = Image.blend(x_image, resized_heatmap_img, alpha=intensity)

            return np.array(saliency_map_img)

    @property
    def last_conv_layer(self):
        """Return the last convolution layer of `self.model.layers`"""
        for layer in self.model.layers[::-1]:
            if isinstance(layer, keras.layers.Conv2D):
                return layer
