import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 15 
DATASET_PATH = 'PlantDataset' 

def custom_preprocess(img):
    return preprocess_input(img)

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,        
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess, 
    validation_split=0.2
)

print("\n--- Loading Training Data ---")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("\n--- Loading Validation Data ---")
validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\nClasses found: {class_names}")

# This tells the model: "Pay 7x more attention to Yucca than Snake Plant"
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

# --- 3. BUILD MODEL ---
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False, 
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
# Keep the brain frozen for this first phase
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) 
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    'best_plant_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# --- 4. TRAIN WITH WEIGHTS ---
print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,  # <--- This fixes the imbalance
    callbacks=[checkpoint, early_stopping]
)

# --- SAVE LABELS & PLOT ---
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print("\nLabels saved.")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('training_results_graph.png')
    plt.show()

plot_history(history)