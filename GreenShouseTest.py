import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIG ---
# 60% is a safer bet. If it's less than 60% sure, it's probably not a plant.
THRESHOLD = 60.0  

# Load model and labels
print("Loading model...")
# FIX 1: Make sure this matches your actual file name!
# We named it 'best_plant_model_deep_tuned.keras' in the last step.
model = tf.keras.models.load_model('best_plant_model_deep_tuned.keras')

try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ['Aloe Vera', 'Areca Palm', 'Snake Plant', 'Yucca']

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Center Crop
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # 2. Resize and Preprocess
    img_resized = cv2.resize(cropped_frame, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # 3. Predict
    predictions = model.predict(img_array, verbose=0)
    
    # FIX 2: REMOVED tf.nn.softmax
    # The model already outputs softmax probabilities. We just grab the array.
    score = predictions[0] 
    
    # Get the best guess
    raw_confidence = 100 * np.max(score)
    best_class = class_names[np.argmax(score)]

    # 4. Logic Check
    if raw_confidence < THRESHOLD:
        label_text = f"Unknown ({raw_confidence:.1f}%)"
        color = (0, 0, 255) # Red
    else:
        label_text = f"{best_class}: {raw_confidence:.1f}%"
        color = (0, 255, 0) # Green

    # 5. Draw
    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame, (start_x, start_y), (start_x+min_dim, start_y+min_dim), (255, 255, 255), 2)

    cv2.imshow('Plant Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()