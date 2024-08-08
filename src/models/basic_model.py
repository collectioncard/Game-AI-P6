from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    # I had chatgpt help me with this because I didnt understand how it worked at first - Thomas Wessel
    # If this is wrong, please change it.
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),

            # Add a dropout layer to prevent overfitting (does 20% work?)
            layers.Dropout(0.3),

            layers. Dense(28, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])


    def _compile_model(self):

        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )