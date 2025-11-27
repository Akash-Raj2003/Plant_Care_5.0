import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('best_plant_model.h5')

# Load all of the class names 
try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ['Aloe Vera', 'Areca Palm', 'Snake Plant', 'Yucca'] 

print(f"Loaded classes: {class_names}")

# 3. Start the Webcam
cap = cv2.VideoCapture(0) # 0 is usually the default laptop camera

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    min_dim = min(h, w)
    
    # Calculate the center square
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Crop the image to be a square
    cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # NOW resize the square crop (No distortion!)
    img_resized = cv2.resize(cropped_frame, (224, 224))
    
    # --- PREPARE FOR MODEL ---
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # --- PREDICT ---
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]).numpy() # Convert to numpy for easier handling

    # --- FIX 2: GET TOP 3 PREDICTIONS ---
    # Sort from highest to lowest confidence
    top_3_indices = score.argsort()[-3:][::-1]
    
    # Display logic
    y_position = 30
    for i in top_3_indices:
        label = class_names[i]
        confidence = score[i] * 100
        
        # Color: Green if > 80%, Yellow if > 50%, Red otherwise
        if confidence > 80:
            color = (0, 255, 0) # Green
        elif confidence > 50:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 0, 255) # Red
            
        text = f"{label}: {confidence:.1f}%"
        
        # Draws the text on the frame on the top-left corner
        cv2.putText(frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_position += 30 # Move down for the next label

    # Draw a box showing where the AI is looking at to classify
    cv2.rectangle(frame, (start_x, start_y), (start_x+min_dim, start_y+min_dim), (255, 255, 255), 2)

    cv2.imshow('Plant Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()