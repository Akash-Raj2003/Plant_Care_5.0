import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

THRESHOLD = 50.0
IMG_SIZE = 224

def custom_preprocess(img):
    return preprocess_input(img)
image_files = [
    'test1.png', 
    'test2.png',
    'test3.png',
    'test4.png', 
    'test5.png'
]

output_folder = 'predicted_images'
os.makedirs(output_folder, exist_ok=True)

print("Loading model...")

model = tf.keras.models.load_model('best_plant_model_balanced.keras')

try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ['Aloe Vera', 'Areca Palm', 'Snake Plant', 'Yucca']

print(f"Model loaded. Processing {len(image_files)} images...")

for file_path in image_files:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        continue
        
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {file_path}")
        continue
    
    h, w, _ = frame.shape

    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    img_array = np.expand_dims(img_resized, axis=0)
    
    img_array = custom_preprocess(img_array)

    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    
    raw_confidence = 100 * np.max(score)
    best_class = class_names[np.argmax(score)]

    if raw_confidence < THRESHOLD:
        label_text = f"Unknown ({raw_confidence:.1f}%)"
        color = (0, 0, 255)
    else:
        label_text = f"{best_class}: {raw_confidence:.1f}%"
        color = (0, 255, 0)

    print(f"Image: {file_path} -> {label_text}")

    cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), 20) 
    
    font_scale = max(1.5, h / 1000.0) 
    thickness = max(2, int(h / 500.0))
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    cv2.rectangle(frame, (20, 20), (20 + text_w + 20, 20 + text_h + 40), (0,0,0), -1) 
    cv2.putText(frame, label_text, (30, 20 + text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    output_filename = f"pred_{os.path.basename(file_path)}"
    save_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(save_path, frame)

    max_display_w = 1200
    max_display_h = 800
    scale = min(max_display_w / w, max_display_h / h, 1.0)
    
    if scale < 1.0:
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        display_frame = frame

    cv2.imshow('Prediction Result', display_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()