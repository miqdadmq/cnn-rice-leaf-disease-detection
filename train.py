# ===============================================
# Import Libraries
# ===============================================
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================
# Download dataset ZIP
# =======================================

!wget https://huggingface.co/datasets/amdmqd/rice_leaf_disease_dataset/resolve/main/rice_leaf_disease_dataset.zip

!unzip rice_leaf_disease_dataset.zip

# =======================================
# Split Train and Test
# =======================================

SOURCE_DIR = 'rice_leaf_disease_dataset'
OUTPUT_DIR = 'rice_leaf_disease_dataset_split'

train_split = 0.7
val_split = 0.15
test_split = 0.15

for split in ['train', 'val', 'test']:
  os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

for class_name in (os.listdir(SOURCE_DIR)):
  class_path = os.path.join(SOURCE_DIR, class_name)
  if not os.path.isdir(class_path):
    continue

  images = os.listdir(class_path)
  random.shuffle(images)

  train_size = int(train_split * len(images))
  val_size = train_size + int(val_split * len(images))

  splits = {
      'train' : images[:train_size],
      'val' : images[train_size:val_size],
      'test' : images[val_size:]
  }

  for split, files in splits.items():
    split_dir = os.path.join(OUTPUT_DIR, split, class_name)
    os.makedirs(split_dir, exist_ok=True)

    for file in files:
      shutil.copy(os.path.join(class_path, file), os.path.join(split_dir, file))

print("[INFO] Split Dataset Into : ", OUTPUT_DIR)

# =======================================
# Image Data Generator
# =======================================

IMG_SIZE = (180, 180)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = train_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =======================================
# CNN Model
# =======================================

model = models.Sequential([
    layers.Input(shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0, 3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =======================================
# Callbacks
# =======================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

callbacks_list = [early_stop, reduce_lr, checkpoint]

# =======================================
# Training
# =======================================

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks_list
)