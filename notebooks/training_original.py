"""
Brain Tumor Detection & Classification
========================================
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma | meningioma | notumor | pituitary

Directory structure expected:
  dataset/
    Training/
      glioma/       *.jpg
      meningioma/   *.jpg
      notumor/      *.jpg
      pituitary/    *.jpg
    Testing/
      glioma/
      meningioma/
      notumor/
      pituitary/

Run:  python notebooks/train_model.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # EfficientNetB0 native input
BATCH_SIZE  = 64
EPOCHS_HEAD = 15           # train only the new head first
EPOCHS_FINE = 30           # then fine-tune top layers of backbone
LR_HEAD     = 1e-3
LR_FINE     = 1e-4
DATASET_DIR = "dataset"
MODEL_OUT   = "backend/model/brain_tumor_model.h5"
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
SEED        = 42

# ─── Reproducibility ─────────────────────────────────────────────────────────
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── GPU / CPU setup ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] Training on {len(gpus)} GPU(s): {gpus}")
else:
    # Maximise CPU throughput
    num_threads = os.cpu_count() or 4
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    print(f"[CPU] No GPU found – using CPU with {num_threads} threads")

# ─── 1. Data generators with augmentation ────────────────────────────────────
print("[1/6] Building data generators …")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.15,          # 15 % of Training used for validation
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.15,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Training"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Training"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Testing"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print(f"  Train samples : {train_gen.samples}")
print(f"  Val   samples : {val_gen.samples}")
print(f"  Test  samples : {test_gen.samples}")
print(f"  Classes       : {train_gen.class_indices}")

# Save class index map for the API
os.makedirs("backend/model", exist_ok=True)
with open("backend/model/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

# ─── 2. Model Architecture (Transfer Learning – EfficientNetB0) ───────────────
print("\n[2/6] Building model …")

def build_model(num_classes: int, img_size: int = IMG_SIZE) -> keras.Model:
    """
    Architecture
    ────────────
    Input (224×224×3)
      └─ EfficientNetB0 (ImageNet weights, frozen initially)
           └─ GlobalAveragePooling2D
                └─ BatchNormalization
                     └─ Dense(512, relu) + Dropout(0.4)
                          └─ Dense(256, relu) + Dropout(0.3)
                               └─ Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Backbone – freeze for head-only training phase
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False  # freeze initially

    # Classification head
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="BrainTumorClassifier")
    return model, backbone


model, backbone = build_model(num_classes=len(CLASSES))
model.summary()

# ─── 3. Phase 1 – Train head only ────────────────────────────────────────────
print("\n[3/6] Phase 1: training head …")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=3, min_lr=1e-7),
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    verbose=1,
)

# ─── 4. Phase 2 – Fine-tune top layers of backbone ───────────────────────────
print("\n[4/6] Phase 2: fine-tuning …")

# Unfreeze top 50 layers of EfficientNetB0
backbone.trainable = True
for layer in backbone.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=4, min_lr=1e-8),
    keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    ),
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    verbose=1,
)

# ─── 5. Evaluation ───────────────────────────────────────────────────────────
print("\n[5/6] Evaluating on test set …")

test_gen.reset()
y_pred_proba = model.predict(test_gen, verbose=1)
y_pred  = np.argmax(y_pred_proba, axis=1)
y_true  = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ─── 6. Plots ────────────────────────────────────────────────────────────────
print("\n[6/6] Saving plots …")
os.makedirs("notebooks/plots", exist_ok=True)

# Merge histories from both phases
def merge(h1, h2, key):
    return h1.history[key] + h2.history[key]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Brain Tumor Classifier – Training History", fontsize=14)

# Accuracy
axes[0].plot(merge(history1, history2, "accuracy"),     label="Train")
axes[0].plot(merge(history1, history2, "val_accuracy"), label="Val")
axes[0].axvline(x=len(history1.history["accuracy"]) - 1,
                color="gray", linestyle="--", label="Fine-tune start")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Loss
axes[1].plot(merge(history1, history2, "loss"),     label="Train")
axes[1].plot(merge(history1, history2, "val_loss"), label="Val")
axes[1].axvline(x=len(history1.history["loss"]) - 1,
                color="gray", linestyle="--", label="Fine-tune start")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/plots/training_history.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig("notebooks/plots/confusion_matrix.png", dpi=150)
plt.close()

print(f"\n✓ Model saved  → {MODEL_OUT}")
print( "✓ Plots saved  → notebooks/plots/")
print( "✓ Class map    → backend/model/class_indices.json")
