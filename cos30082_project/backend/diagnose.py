import os, numpy as np
import tensorflow as tf, builtins
builtins.tf = tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
keras.config.enable_unsafe_deserialization()

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
import json
emotion_model = keras.models.load_model(os.path.join(MODELS, 'best_emotion_model.keras'))
with open(os.path.join(MODELS, 'emotion_labels.json')) as f:
    emo = {int(k): v for k, v in json.load(f).items()}
EMO_LABELS = [emo[i] for i in range(len(emo))]
SIZE = emotion_model.input_shape[1]

print("Input shape :", emotion_model.input_shape)
print("Output shape:", emotion_model.output_shape)
print("Labels      :", EMO_LABELS)
print("\n--- random-noise test (should vary, not all 'angry') ---")
for i in range(5):
    fake = (np.random.rand(1, SIZE, SIZE, 3) * 255).astype('float32')
    pred = emotion_model.predict(preprocess_input(fake), verbose=0)[0]
    print(f"  {i}: {[f'{p:.2f}' for p in pred]} -> {EMO_LABELS[int(np.argmax(pred))]}")