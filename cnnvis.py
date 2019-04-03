import numpy as np
import matplotlib.pyplot as plt
import keras
from prettytable import PrettyTable


class Visualizer:

    def __init__(self,
                 model=None,
                 model_input_shape=(224, 224, 3),
                 training_set_info='1000 Categories of ImageNet'):
        """
        Initializes a CNNVisualizer.
        The convolution base of a VGG16 network (trained on ImageNet) is the default model.

        :param model:
            if model is None, the vgg16 model is used;
            if model is a keras.models.Sequential or keras.models.Model instance, the passed-in model is used
        :param model_input_shape: only relevant if `model` is None, that is, when the vgg16 model is used
        """
        if model == None:
            self.model = keras.applications.VGG16(weights='imagenet', include_top=True, input_shape=model_input_shape)
        else:
            if isinstance(model, keras.models.Sequential) or isinstance(model, keras.models.Model):
                self.model = model
            else:
                raise ValueError('The model passed in through the `model` parameter is neither \
                    a keras.models.Sequential instance nor a keras.models.Model instance.')

        self.training_set_info = training_set_info

    def model_summary(self, style='cnn', input_shape=(1, 224, 224, 3)):
        """
        Prints out the model summary of `self.model`.

        :param stype:
            if style = 'default', prints out the default `self.model.summary()`
            if style = 'cnn', prints out the cnn style summary
        :param input_shape: the input shape (including batch size) of `self.model`
        """
        if style == 'default':
            print(self.model.summary())
        elif style == 'cnn':
            print('CNN Style Model Summary')

            num_Conv2D = 0
            num_MaxPooling2D = 0
            num_Dense = 0

            summary = PrettyTable()
            summary.field_names = ["Layer Name",
                                   "Layer Type",
                                   "Kernel Num",
                                   "Kernel Size",
                                   "Kernel Padding",
                                   "Output Shape"]

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
            print('Training set: {}'.format(self.training_set_info))
            print('Number of Conv2D layers: {}'.format(num_Conv2D))
            print('Number of MaxPooling2D layers: {}'.format(num_MaxPooling2D))
            print('Number of Dense layers: {}'.format(num_Dense))

        else:
            raise ValueError('{} is not a valid value for keyword argument "style".'.format(style))

    def get_kernels(self, layer_name, style='tensors'):
        """
        Returns the convolution kernels / weights of a layer.
        The shape of the returned kernels is (num_kernels, height, width)

        :param layer_name: the name of the layer whose convolution kernels will be returned
        :return: the convolution kernels of the layer with name `layer_name`
        """
        kernels = self.model.get_layer(layer_name).get_weights()[0]

        if style == 'tensors':
            return self.index_last_tensor_to_index_first_tensor(kernels)  # for easy indexing
        elif style == 'plots':
            self.plot_depth_last_tensor(kernels, 'imshow')
        elif style == 'both':
            self.plot_depth_last_tensor(kernels, 'imshow')
            return self.index_last_tensor_to_index_first_tensor(kernels)  # for easy indexing
        else:
            raise ValueError('{} is not a valid value for keyword argument `output`.'.format(style))

    def get_output_of_layer(self, layer_name):
        """
        Returns symbolic output of a 2D convolution layers.
        Helper method of `self.get_model_with_output_layer` method.

        References:
        * https://keras.io/layers/about-keras-layers/
        * https://keras.io/layers/convolutional/
        """
        layer = self.model.get_layer(layer_name)  # captures 'layer not found' errors
        if isinstance(layer, keras.layers.Conv2D):
            return layer.output
        else:
            raise ValueError('Layer with name {} is not a 2D convolution layer.'.format(layer_name))

    def get_model_with_output_layer(self, layer_name):
        """
        Returns a keras.models.Model that inputs an image and
        outputs activations of a specific 2D convolution layer.
        Helper method of 'self.get_feature_maps_to_image' method.

        References:
        * https://keras.io/models/model/
        """
        return keras.models.Model(inputs=self.model.input,
                                  outputs=self.get_output_of_layer(layer_name))

    @staticmethod
    def img_path_to_img_tensor(img_path):
        """
        Loads a JPEG or PNG image from a path to a normalized tensor.
        Helper method of 'self.get_feature_maps_to_image' method.

        :param img_path: the path of the JPEG or PNG image
        :return: a normalized (values: 0 ~ 1) image (numpy) tensor
        """
        # maybe get the input shape of the model here
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_tensor = keras.preprocessing.image.img_to_array(img)  # img_tensor.shape = (150, 150, 3)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # img_tensor.shape = (1, 150, 150, 3)

        # the shape of a batch of n images = (n, image_height, image_width, channels)
        # even a single image needs to be in batch shape format when inputted to a model

        return img_tensor / 255.

    @staticmethod
    def plot_depth_last_tensor(feature_maps_tensor, style):
        """
        Plot all feature maps in the feature maps tensor in matplotlib.
        Helper method of 'self.get_feature_maps_to_image' method.

        :param feature_maps_tensor: A tensor of feature maps
        """
        num_feature_maps = feature_maps_tensor.shape[3]
        width_of_square_matshow_array = int(num_feature_maps ** 0.5 + 0.5)

        fig, _ = plt.subplots(width_of_square_matshow_array,
                              width_of_square_matshow_array,
                              figsize=(10, 10))
        axes = fig.get_axes()

        for i in np.arange(0, num_feature_maps):
            ax = axes[i]
            img = feature_maps_tensor[:, :, :, i]
            if style == 'matshow':
                ax.matshow(img)
            elif style == 'imshow':
                ax.imshow(np.sum(img, axis=2))  # all channels are summed together and normalized
            else:
                raise ValueError('{} is not a valid value for `style`.'.format(style))
            ax.axis('off')
            ax.set_title('{0}'.format(axes.index(ax) + 1))

        for i in np.arange(num_feature_maps, width_of_square_matshow_array ** 2):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def index_last_tensor_to_index_first_tensor(img_tensor):
        index_index = 3

        feature_maps = []
        for i in range(img_tensor.shape[index_index]):
            feature_maps.append([img_tensor[:, :, :, i]])

        return np.concatenate(feature_maps, axis=0)

    def get_feature_maps(self, layer_name, img_path, style='tensors'):
        """
        Plot and/or return the feature maps outputted by a layer during prediction.
        'Layer not found' errors are captured in method 'get_output_of layer'.

        :param layer_name: the name of the layer whose output you want to capture, type: str
        :param img_path: the path to the jpg or png file, type: str
        :param output_style:
            if output_style='plots', feature maps are plotted using matplotlib;
            if output_style='tensors', feature maps are returned as 4D numpy arrays;
            if output_style='both', feature maps are both plotted and returned
        :return: refer to the description of param 'output_style'
        """

        img_tensor = self.img_path_to_img_tensor(img_path)  # jpg to tensor

        # construct a model with output at the layer with name 'layer_name'
        #     and capture the output of that layer during prediction
        model_with_output_layer_layer_name = self.get_model_with_output_layer(layer_name)
        # feature_maps.shape is (1, height, width, num_feature_maps)
        feature_maps = model_with_output_layer_layer_name.predict(img_tensor)

        if style == 'tensors':
            # 'depth_first_tensor_to_depth_last_tensor' method is used so that
            #     the returned tensor is easier to work with using indexing
            # new feature_maps.shape = (num_feature_maps, height, width)
            return self.index_last_tensor_to_index_first_tensor(feature_maps)
        elif style == 'plots':
            self.plot_depth_last_tensor(feature_maps, 'matshow')
        elif style == 'both':
            self.plot_depth_last_tensor(feature_maps, 'matshow')
            # 'depth_first_tensor_to_depth_last_tensor' method is used so that
            # the returned tensor is easier to work with using indexing
            return self.index_last_tensor_to_index_first_tensor(feature_maps)
        else:
            raise ValueError('{} is not a valid value for keyword argument `output`.'.format(style))

    def get_mean_activations(self, layer_name, img_path):
        """
        Returns the mean activations of a specific layer to an image.

        :param layer_name: the name of the specific layer
        :param img_path: the path to the JPEG or PNG file
        """
        if isinstance(self.model.get_layer(layer_name), keras.layers.Conv2D):
            img_tensor = self.img_path_to_img_tensor(img_path)

            model_with_output_layer_layer_name = self.get_model_with_output_layer(layer_name)
            feature_maps = model_with_output_layer_layer_name.predict(img_tensor)
            feature_maps = self.index_last_tensor_to_index_first_tensor(feature_maps)

            # now feature_maps is an index first tensor
            # feature_maps.shape = (num_kernels, height, width, depth)

            num_kernels = feature_maps.shape[0]

            mean_activations = []
            for i in range(num_kernels):
                mean_activations.append(np.mean(feature_maps[i]))

            return np.arange(0, num_kernels, num_kernels // 10), mean_activations
        else:
            raise ValueError('The layer with name {} is not a 2D convolution layer.')

    def get_max_activation_image(self, layer_name, kernel_index):
        pass

    def get_saliency_map(self, img_path):
        pass
