from keras.initializers import initializers

from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam


class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Using the model that we are submitting for part 6
        base_model = models.load_model('results/basic_model_22_epochs_timestamp_1723141018.keras')

        for layer in base_model.layers:
            layer.trainable = False

        self._randomize_layers(base_model)

        # Remove the final softmax layer
        base_model = models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        # Create a new model and add the base model
        self.model = Sequential()

        self.model.add(base_model)

        # Add new fully connected layers
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(categories_count, activation='softmax'))

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    @staticmethod
    def _randomize_layers(model):

        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                layer.kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=0.8)
                layer.bias_initializer = initializers.RandomNormal(mean=0.0, stddev=0.8)
                layer.build(layer.input_shape)
