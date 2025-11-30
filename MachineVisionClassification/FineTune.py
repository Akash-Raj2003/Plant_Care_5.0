import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# We import the function, but wrap it to prevent Windows errors
from tensorflow.keras.applications.efficientnet import preprocess_input 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# config
DATASET_PATH = 'PlantDataset'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 30 

# wrap preprocess_input to avoid Windows issues
def custom_preprocess(img):
    return preprocess_input(img)

# data augmentation and generators for training and validation
print("\n--- Preparing Data Generators ---")
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    validation_split=0.2,
    rotation_range=40,       # Rotate up to 40 degrees
    width_shift_range=0.3,   # Move left/right
    height_shift_range=0.3,  # Move up/down
    shear_range=0.2,         # Perspective distortion
    zoom_range=0.3,          # Zoom in significantly
    brightness_range=[0.7, 1.3], # Darken/Lighten for webcam realism
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

# Save class names for later
class_names = list(train_generator.class_indices.keys())
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"Classes found: {class_names}")

# model loading and fine-tuning
print("\n--- Loading Previous Model ---")
try:
    model = tf.keras.models.load_model('best_plant_model.h5')
    print("Loaded 'best_plant_model.h5' successfully.")
except:
    print("Could not find 'best_plant_model.h5'. Please ensure the file exists.")
    exit()

# Unlock the model for Deep Fine-Tuning
model.trainable = True 
print(f"Total layers: {len(model.layers)}")

# Unfreeze the top 60 layers to allow fine-tuning
for layer in model.layers[:-60]:
    layer.trainable = False
print("Top 60 layers unfrozen.")

# compile the model with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5), # Very slow learning rate to avoid large weight updates
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
checkpoint = ModelCheckpoint(
    'best_plant_model_deep_tuned.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)
# Stop training if no improvement for 8 epochs
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True
)

# If accuracy stalls for 3 epochs, cut the learning rate in half
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# train the model
print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS, 
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

# plot training results
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))

    # Graph 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.grid(True)

    # Graph 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.grid(True)

    # Save the graph to disk
    plt.savefig('training_results_graph.png')
    print("\nGraph saved as 'training_results_graph.png'")
    
    # Show the graph
    plt.show()

# Run the plotter
plot_results(history)
print("\nDeep Fine-Tuning Complete. Model saved as 'best_plant_model_deep_tuned.keras'")