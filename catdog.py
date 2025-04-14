from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
DATASET_PATH = '../datasets/dogs_vs_cats'
DATASET_TRAIN = DATASET_PATH + '/train/PetImages'
DATASET_TEST = DATASET_PATH + '/test'

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_validation_data_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2],
    zoom_range=0.2,
    validation_split=0.2
)

testing_data_gen=ImageDataGenerator(
    rescale=1./255
)
# Load train (70%) set
train_ds=training_validation_data_gen.flow_from_directory(
    DATASET_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED,
    shuffle=True,
   #color_mode='grayscale'
)

val_ds=training_validation_data_gen.flow_from_directory(
    DATASET_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=True,
    #color_mode='grayscale'
)

test_ds=testing_data_gen.flow_from_directory(
    DATASET_TEST,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED,
    shuffle=True,
    #color_mode='grayscale'
)
# Build the ANN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('CatDogANN.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
EPOCHS = 50
history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs = EPOCHS,
  callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")