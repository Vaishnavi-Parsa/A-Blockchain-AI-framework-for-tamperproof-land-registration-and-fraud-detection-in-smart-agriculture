from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from lightgbm import LGBMClassifier

class CNN_2D:
    def __init__(self, model_type=None):
        self.model_type = model_type
        self.model = None

    # ---------------------- CNN Model ---------------------- #
    def build_cnn(self, input_shape=(64, 64, 3), num_classes=2):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2,2),

            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # ---------------------- LightGBM Default ---------------------- #
    def build_lightgbm(self):
        return LGBMClassifier(learning_rate=0.1)

    # ---------------------- Model Selector ---------------------- #
    def get_model(self):
        if self.model_type is not None and self.model_type.lower() == "cnn":
            print("🔥 Selected Model: CNN")
            self.model = self.build_cnn()
        else:
            print("⚡ Default Model: LightGBM")
            self.model = self.build_lightgbm()

        return self.model
