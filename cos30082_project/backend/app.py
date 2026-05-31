# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL BACKEND  —  backend/app.py
#
#  Run locally in VS Code (no Colab, no ngrok). Loads your trained models from
#  ../models and serves the same JSON schema your app.js already expects.
#
#  Run:
#     cd backend
#     pip install flask flask-cors tensorflow opencv-python pillow numpy
#     python app.py
#
#  Then point app.js:  const BACKEND_URL = 'http://localhost:5000';
# ══════════════════════════════════════════════════════════════════════════════

import os, io, base64, json, time, threading
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import builtins
builtins.tf = tf                      # lets Lambda layers reload cleanly
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
keras.config.enable_unsafe_deserialization()

# ── Paths: models live in ../models relative to this file ─────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'models')
EMPLOYEE_DB_PATH = os.path.join(MODELS, 'employee_database.json')

# Built-in OpenCV face detector (ships with opencv-python)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame_rgb):
    """Detect the largest face and return a tight crop (with margin). Falls back to full frame."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame_rgb, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
    m = int(0.2 * w)                                      # 20% margin
    y1, y2 = max(0, y - m), min(frame_rgb.shape[0], y + h + m)
    x1, x2 = max(0, x - m), min(frame_rgb.shape[1], x + w + m)
    return frame_rgb[y1:y2, x1:x2], True

def _l2(t):
    return tf.math.l2_normalize(t, axis=1)

# ── Load all models from disk (once, at startup — no training) ────────────────
print('[..] Loading models from', MODELS)
face_embedder = keras.models.load_model(
    os.path.join(MODELS, 'face_cls_11785.h5'),
    compile=False, safe_mode=False, custom_objects={'_l2': _l2})
emotion_model  = keras.models.load_model(os.path.join(MODELS, 'best_emotion_model.keras'))
liveness_model = keras.models.load_model(os.path.join(MODELS, 'best_liveness_model.keras'))

with open(os.path.join(MODELS, 'emotion_labels.json')) as f:
    emo_idx_to_label = {int(k): v for k, v in json.load(f).items()}
EMO_LABELS = [emo_idx_to_label[i] for i in range(len(emo_idx_to_label))]

# In-memory databases
watchlist     = {}        # suspect_name -> normalized embedding

wl_path = os.path.join(MODELS, 'criminal_watchlist.json')
if os.path.exists(wl_path):
    with open(wl_path) as f:
        watchlist = {k: np.array(v, dtype='float32') for k, v in json.load(f).items()}
print(f'[ok] Models loaded | watchlist suspects: {len(watchlist)}')

# Input sizes (read from the models themselves, fall back to known values)
EMO_SIZE  = emotion_model.input_shape[1]  or 96
FACE_SIZE = face_embedder.input_shape[1]  or 96
LIVE_SIZE = liveness_model.input_shape[1] or 224
print(f'[cfg] emotion={EMO_SIZE}px  face={FACE_SIZE}px  liveness={LIVE_SIZE}px')
print(f'[cfg] emotion labels: {EMO_LABELS}')

EMOJI = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😊',
         'neutral':'😐','sad':'😢','surprise':'😲'}

IDENTITY_THRESHOLD = 0.6
WATCHLIST_THRESHOLD = 0.85

# ── Helpers ───────────────────────────────────────────────────────────────────
def b64_to_rgb(data_url):
    """Browser sends 'data:image/jpeg;base64,...' → RGB uint8 numpy array."""
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data_url))).convert('RGB')
    return np.array(img)

def embed(frame_rgb):
    face_in = preprocess_input(cv2.resize(frame_rgb, (FACE_SIZE, FACE_SIZE)).astype('float32'))
    e = face_embedder.predict(np.expand_dims(face_in, 0), verbose=0)[0]
    return e / (np.linalg.norm(e) + 1e-8)

def save_employee_db():
    """Persist the employee database (embedding + employee_id) to disk."""
    data = {name: {"embedding": rec["embedding"].tolist(),
                   "employee_id": rec["employee_id"]}
            for name, rec in face_database.items()}
    with open(EMPLOYEE_DB_PATH, 'w') as f:
        json.dump(data, f)
    print(f"[ok] Saved {len(data)} employees to disk")

def load_employee_db():
    """Load the employee database from disk if it exists."""
    if os.path.exists(EMPLOYEE_DB_PATH):
        with open(EMPLOYEE_DB_PATH) as f:
            data = json.load(f)
        loaded = {name: {"embedding": np.array(rec["embedding"], dtype='float32'),
                         "employee_id": rec["employee_id"]}
                  for name, rec in data.items()}
        print(f"[ok] Loaded {len(loaded)} employees: {list(loaded.keys())}")
        return loaded
    return {}

def screen_watchlist(emb, threshold=WATCHLIST_THRESHOLD):
    if not watchlist:
        return {'flagged': False, 'match_id': None}
    names = list(watchlist.keys())
    embs = np.vstack([watchlist[n] for n in names])
    sims = embs @ emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(emb) + 1e-8)
    idx = int(np.argmax(sims)); score = float(sims[idx])
    if score >= threshold:
        return {'flagged': True, 'match_id': names[idx], 'confidence': score}
    return {'flagged': False, 'match_id': None, 'confidence': score}



# ── Single-frame pipeline: liveness → identity → emotion → watchlist ──────────
# def run_pipeline(frame_rgb, use_liveness=True):
    result = {
        "state": "UNKNOWN",          # SUCCESS | SPOOF | UNKNOWN
        "name": "Unknown",
        "employee_id": "—",
        "emotion_label": "—",
        "emotion_icon": "—",
        "liveness": "UNKNOWN",
        "liveness_conf": 0.0,
        "identity_conf": 0.0,
        "criminal_flag": False,
    }
    face_crop, found = crop_face(frame_rgb)
    if not found:
        result["state"] = "UNKNOWN"
        result["txt_detail"] = "No face detected"
    # ---- Liveness gate ----
    # IMPORTANT: this model scores REAL faces LOW, so live == prob < 0.5
    live_in = preprocess_input(cv2.resize(frame_rgb, (LIVE_SIZE, LIVE_SIZE)).astype('float32'))
    live_prob = float(liveness_model.predict(np.expand_dims(live_in, 0), verbose=0)[0][0])
    is_live = live_prob < 0.5
    result["liveness_conf"] = round((1.0 - live_prob) * 100, 1)
    result["liveness"] = "REAL" if is_live else "SPOOF"

    if use_liveness and not is_live:
        result["state"] = "SPOOF"
        return result

    # ---- Face embedding + identity match ----
    emb = embed(frame_rgb)
    best_name, best_score = "Unknown", -1.0
    for name, db_emb in face_database.items():
        s = float(np.dot(emb, db_emb))
        if s > best_score:
            best_name, best_score = name, s
    result["identity_conf"] = round(best_score * 100, 1)

    if best_score >= IDENTITY_THRESHOLD:
        result["name"] = str(best_name)
        result["employee_id"] = f"EMP-{abs(hash(best_name)) % 9000 + 1000}"
        result["state"] = "SUCCESS"
    else:
        result["state"] = "UNKNOWN"

    # ---- Emotion ----
    emo_in = preprocess_input(cv2.resize(frame_rgb, (EMO_SIZE, EMO_SIZE)).astype('float32'))
    emo_pred = emotion_model.predict(np.expand_dims(emo_in, 0), verbose=0)[0]
    emo_label = EMO_LABELS[int(np.argmax(emo_pred))]
    result["emotion_label"] = emo_label.capitalize()
    result["emotion_icon"] = EMOJI.get(emo_label, "🙂")

    # ---- Criminal watchlist (innovation) ----
    chk = screen_watchlist(emb)
    result["criminal_flag"] = bool(chk["flagged"])
    if chk["flagged"]:
        result["state"] = "SPOOF"   # reuse red-alert UI state for a watchlist hit
        result["name"] = f"WATCHLIST MATCH: {chk['match_id']}"

    return result

face_database = load_employee_db()
def run_pipeline(frame_rgb, use_liveness=True):
    result = {
        "state": "UNKNOWN",          # SUCCESS | SPOOF | UNKNOWN
        "name": "Unknown",
        "employee_id": "—",
        "emotion_label": "—",
        "emotion_icon": "—",
        "liveness": "UNKNOWN",
        "liveness_conf": 0.0,
        "identity_conf": 0.0,
        "criminal_flag": False,
    }

    # ---- Crop to the detected face first (model was trained on face crops) ----
    face_crop, found = crop_face(frame_rgb)
    if not found:
        # No face in frame — tell the UI and stop
        result["state"] = "UNKNOWN"
        result["name"] = "No face detected"
        return result
    frame_rgb = face_crop   # everything below now uses the tight face crop

    # ---- Liveness gate (this model scores REAL faces LOW, so live == prob < 0.5) ----
    live_in = preprocess_input(cv2.resize(frame_rgb, (LIVE_SIZE, LIVE_SIZE)).astype('float32'))
    live_prob = float(liveness_model.predict(np.expand_dims(live_in, 0), verbose=0)[0][0])
    is_live = live_prob < 0.5
    result["liveness_conf"] = round((1.0 - live_prob) * 100, 1)
    result["liveness"] = "REAL" if is_live else "SPOOF"

    if use_liveness and not is_live:
        result["state"] = "SPOOF"
        return result

    # ---- Face embedding + identity match ----
    emb = embed(frame_rgb)
    best_name, best_score, best_id = "Unknown", -1.0, "—"
    for name, rec in face_database.items():
        s = float(np.dot(emb, rec["embedding"]))
        if s > best_score:
            best_name, best_score, best_id = name, s, rec["employee_id"]
    result["identity_conf"] = round(best_score * 100, 1)

    if best_score >= IDENTITY_THRESHOLD:
        result["name"] = str(best_name)
        result["employee_id"] = best_id         
        result["state"] = "SUCCESS"
    else:
        result["state"] = "UNKNOWN"

    # ---- Emotion ----
    emo_in = preprocess_input(cv2.resize(frame_rgb, (EMO_SIZE, EMO_SIZE)).astype('float32'))
    emo_pred = emotion_model.predict(np.expand_dims(emo_in, 0), verbose=0)[0]
    emo_label = EMO_LABELS[int(np.argmax(emo_pred))]
    result["emotion_label"] = emo_label.capitalize()
    result["emotion_icon"] = EMOJI.get(emo_label, "🙂")

    # ---- Criminal watchlist (innovation) ----
    chk = screen_watchlist(emb)
    result["criminal_flag"] = bool(chk["flagged"])
    if chk["flagged"]:
        result["state"] = "SPOOF"   # reuse red-alert UI state for a watchlist hit
        result["name"] = f"WATCHLIST MATCH: {chk['match_id']}"

    print(f"[debug] live_prob={live_prob:.3f} is_live={is_live} "
          f"best={best_name} score={best_score:.3f} "
          f"watchlist_flag={result['criminal_flag']} state={result['state']}")
    return result

    return result
def register_face(frame_rgb, name, employee_id):
    face_crop, found = crop_face(frame_rgb)
    if found:
        frame_rgb = face_crop
    face_database[name] = {"embedding": embed(frame_rgb), "employee_id": employee_id}
    save_employee_db()
    print(f"[debug] REGISTERED '{name}' ({employee_id}) — total {len(face_database)}")
    return {"ok": True, "name": name, "employee_id": employee_id,
            "db_size": len(face_database)}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True, "db_size": len(face_database),
                    "employees": list(face_database.keys()),
                    "watchlist": len(watchlist)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    frame = b64_to_rgb(data['image'])
    use_liveness = data.get('use_liveness', True)
    return jsonify(run_pipeline(frame, use_liveness=use_liveness))

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    frame = b64_to_rgb(data['image'])
    out = register_face(frame, data.get('name', 'Unknown'), data.get('employee_id', '—'))
    return jsonify(out)

@app.route('/delete_employee', methods=['POST'])
def delete_employee():
    name = request.get_json().get('name')
    if name in face_database:
        del face_database[name]
        save_employee_db()
        print(f"[debug] DELETED '{name}' — total {len(face_database)}")
        return jsonify({"ok": True, "remaining": len(face_database)})
    return jsonify({"ok": False, "error": "not found"})

if __name__ == '__main__':
    print('[ok] Backend running on http://localhost:5000')
    app.run(host='0.0.0.0', port=5001, debug=False)
