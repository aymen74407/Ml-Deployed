"""
CNN TRAINING SCRIPT -- Run this ONE TIME only

What it does:
  1. Loads images from BOTH Roboflow datasets (train/ splits)
  2. Merges them and de-duplicates by file hash to avoid overlap
  3. Trains a MobileNetV2-based CNN (Transfer Learning)
  4. Saves the trained model to:
       skin-ml-app/models/saved_cnn/skin_type_cnn.h5

After running this script once, the Flask app will load the saved
model instantly on every startup -- no re-training needed.

Usage:
  cd <project-root>
  python skin-ml-app/train_cnn.py
"""

import os
import sys
import time
import glob
import random
import hashlib
import json

import numpy as np

# -- Guard against running this by accident inside Flask ----------------------
if os.environ.get("FLASK_RUN_FROM_CLI"):
    print("[train_cnn] ERROR: Do not run this as part of Flask. "
          "Run it standalone: python skin-ml-app/train_cnn.py")
    sys.exit(1)

print("=" * 70)
print("  SKIN-TYPE CNN TRAINER -- TensorFlow / Keras")
print("=" * 70)

# -- 1. Import TensorFlow (helpful error if not installed) --------------------
print("\n[1/6] Importing TensorFlow ...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"      TensorFlow {tf.__version__} loaded OK")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"      GPUs available: {len(gpus)}  -> {gpus}")
except ImportError:
    print("\n  ERROR: TensorFlow is not installed!")
    print("         Run:  pip install tensorflow")
    sys.exit(1)

# -- 2. Configuration ---------------------------------------------------------
CLASSES      = ['combination', 'dry', 'normal', 'oily']
IMG_SIZE     = 128       # 128x128 RGB -- good balance of speed vs accuracy
BATCH_SIZE   = 32
EPOCHS_FINE  = 20        # fine-tune epochs (early-stopping will cut this short)
SEED         = 42

# Dataset roots -- adjust if your Downloads folder is on a different drive
DATASET_DIRS = [
    r"C:\Users\hamma\Downloads\skin type.v1i.folder",
    r"C:\Users\hamma\Downloads\skin type.v1i.folder (2)",
]

# Where the trained model will be saved (inside the Flask app)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, "models", "saved_cnn")
MODEL_PATH = os.path.join(SAVE_DIR, "skin_type_cnn.h5")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -- 3. Collect image paths from both datasets --------------------------------
print("\n[2/6] Scanning datasets ...")

# Use a dict keyed on MD5 hash of first 8 KB to deduplicate images that appear
# in BOTH dataset folders (same image, different path).
seen_hashes = {}          # hash -> filepath
class_images = {c: [] for c in CLASSES}

for ds_root in DATASET_DIRS:
    train_dir = os.path.join(ds_root, "train")
    if not os.path.isdir(train_dir):
        print(f"      WARNING: Skipping (no train/ found): {ds_root}")
        continue
    print(f"      Scanning: {ds_root}")
    for cls in CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"         WARNING: Class folder missing: {cls_dir}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for fp in glob.glob(os.path.join(cls_dir, ext)):
                with open(fp, "rb") as fh:
                    quick_hash = hashlib.md5(fh.read(8192)).hexdigest()
                if quick_hash not in seen_hashes:
                    seen_hashes[quick_hash] = fp
                    class_images[cls].append(fp)

total = sum(len(v) for v in class_images.values())
print("\n      Class distribution (after dedup):")
for cls in CLASSES:
    print(f"        {cls:>12s}: {len(class_images[cls]):4d} images")
print(f"        {'TOTAL':>12s}: {total:4d} images")

if total < 100:
    print("\n  ERROR: Not enough images found. Check DATASET_DIRS paths above.")
    sys.exit(1)

# -- 4. Build merged, shuffled dataset with train/val split ------------------
print("\n[3/6] Building TensorFlow datasets ...")

all_paths, all_labels = [], []
for label_idx, cls in enumerate(CLASSES):
    paths = class_images[cls][:]
    random.shuffle(paths)
    all_paths.extend(paths)
    all_labels.extend([label_idx] * len(paths))

combined = list(zip(all_paths, all_labels))
random.shuffle(combined)
all_paths, all_labels = zip(*combined)
all_paths  = list(all_paths)
all_labels = list(all_labels)

split = int(0.85 * len(all_paths))
train_paths, val_paths   = all_paths[:split],  all_paths[split:]
train_labels, val_labels = all_labels[:split], all_labels[split:]

print(f"      Train samples : {len(train_paths)}")
print(f"      Val   samples : {len(val_paths)}")

# -- TF dataset pipeline ------------------------------------------------------
def load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_dataset(paths, labels, do_augment=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if do_augment:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(512, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, do_augment=True)
val_ds   = make_dataset(val_paths,   val_labels,   do_augment=False)

# -- 5. Build model (MobileNetV2 backbone -- Transfer Learning) ---------------
print("\n[4/6] Building MobileNetV2 model ...")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False    # freeze pretrained layers first

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.40)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.30)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"      Trainable params: {model.count_params():,}")

# -- 5a. Phase 1 -- Train the head only (frozen backbone) --------------------
print("\n[5/6] Training ...")
print("      -- Phase 1: Head-only training (5 epochs) --")

callbacks_p1 = [
    keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_accuracy"
    ),
]

t0 = time.time()
model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=callbacks_p1,
    verbose=1,
)

# -- 5b. Phase 2 -- Unfreeze top 50 base layers for fine-tuning --------------
print(f"\n      -- Phase 2: Fine-tuning top 50 base layers ({EPOCHS_FINE} epochs max) --")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.4, patience=3, min_lr=1e-6, monitor="val_accuracy"
    ),
]

model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=val_ds,
    callbacks=callbacks_p2,
    verbose=1,
)

elapsed = time.time() - t0
print(f"\n      Training complete in {elapsed/60:.1f} minutes")

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"      Final val accuracy : {acc*100:.1f}%")
print(f"      Final val loss     : {loss:.4f}")

# -- 6. Save model and metadata ----------------------------------------------
print(f"\n[6/6] Saving model to: {MODEL_PATH}")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(MODEL_PATH)

meta = {
    "classes":  CLASSES,
    "img_size": IMG_SIZE,
    "val_acc":  round(float(acc), 4),
    "trained_on_samples": len(train_paths) + len(val_paths),
}
meta_path = os.path.join(SAVE_DIR, "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("      model saved  -> " + MODEL_PATH)
print("      meta.json    -> " + meta_path)
print()
print("=" * 70)
print("  DONE! The Flask app will now load this model instantly.")
print("  val_acc = {:.1f}%  |  trained on {} samples".format(
    acc * 100, len(train_paths) + len(val_paths)))
print("=" * 70)
