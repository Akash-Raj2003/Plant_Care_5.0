import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight 

# --- CONFIGURATION ---
DATASET_PATH = 'PlantDataset'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 30 
# Note: The initial model created best_plant_model.h5 is the STARTING point for this script.

# --- WINDOWS WRAPPER ---
def custom_preprocess(img):
    return preprocess_input(img)

# --- 1. DATA GENERATORS (Moderate Augmentation for Stability) ---
print("\n--- Preparing Data Generators ---")
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,          # Moderate zoom to see trunks/stems
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess, 
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())

# --- 2. CALCULATE CLASS WEIGHTS ---
# This ensures the model maintains fairness against the 400 vs 60 imbalance.
print("\n--- Calculating Class Weights ---")
train_labels = train_generator.classes 
class_indices = np.unique(train_labels)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=class_indices,
    y=train_labels
)
class_weights = dict(zip(class_indices, weights))

print("Class Weights Calculated:")
for idx, weight in class_weights.items():
    name = class_names[idx]
    print(f"  {name}: {weight:.2f}")

# --- 3. MODEL LOADING & DEEP UNFREEZING ---
print("\n--- Loading Balanced Model for Fine-Tuning ---")
try:
    # Load the model saved from the balanced initial training phase
    model = tf.keras.models.load_model('best_plant_model.h5') 
except:
    print("Could not find 'best_plant_model.h5'. Please run the initial training script first.")
    exit()

# Unlock the whole model
model.trainable = True 

# Freeze all layers EXCEPT the last 60 (Deep Fine-Tuning)
print(f"Total layers: {len(model.layers)}")
for layer in model.layers[:-60]:
    layer.trainable = False
print("Top 60 layers unfrozen for Deep Fine-Tuning.")

# --- 4. COMPILE ---
model.compile(
    optimizer=Adam(learning_rate=1e-5), # Tiny LR is CRUCIAL for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. CALLBACKS ---
checkpoint = ModelCheckpoint(
    'best_plant_model_balanced.keras', # FINAL output model name
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, 
    patience=3, 
    min_lr=1e-7,
    verbose=1
)

# --- 6. TRAIN ---
print("\n--- Starting Balanced Fine-Tuning ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS, 
    class_weight=class_weights, # Fixes the bias during training
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

# --- 7. PLOT RESULTS ---
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.grid(True)
    
    plt.savefig('FineTune_Balanced_graph.png')
    plt.show()

plot_results(history)
print("\nDeep Fine-Tuning Complete. Model saved as 'best_plant_model_balanced.keras'")