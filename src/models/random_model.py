from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Using the model that we are submitting for part 6
        base_model = models.load_model('results/basic_model_15_epochs_timestamp_1722977992.keras')

        for layer in base_model.layers:
            layer.trainable = False

        self._randomize_layers(base_model)

        # Remove the final softmax layer
        base_model = models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        # Create a new model and add the base model
        self.model = Sequential()

        self.model.add(base_model)

        # Add new fully connected layers
        self.model.add(layers.Dense(20, activation='relu'))
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
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                layer.set_weights([
                    layer.set_weights([
                        layer.kernel_initializer(shape=layer.kernel.shape),
                        layer.bias_initializer(shape=layer.bias.shape)
                    ])
                ])