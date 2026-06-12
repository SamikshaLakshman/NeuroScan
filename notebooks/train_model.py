"""
Brain Tumor Detection & Classification
======================================
CPU Optimized + Correct EfficientNet Preprocessing
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
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20

LR_HEAD = 1e-3
LR_FINE = 1e-5

DATASET_DIR = "dataset"

MODEL_OUT = "backend/model/brain_tumor_model.h5"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

SEED = 42

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# CPU OPTIMIZATION
# ─────────────────────────────────────────────────────────────

print("\n[CPU MODE] Running on CPU only...")

tf.config.set_visible_devices([], "GPU")

num_threads = os.cpu_count() or 4

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

print(f"Using {num_threads} CPU threads")

# ─────────────────────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────────────────────

print("\n[1/6] Building data generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    # LIGHT AUGMENTATION FOR MRI
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,

    # IMPORTANT: NO HORIZONTAL FLIP
    horizontal_flip=False,

    validation_split=0.15,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.15,
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# TRAIN
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Training"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED,
)

# VALIDATION
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Training"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

# TEST
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Testing"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print(f"\nTrain samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
print(f"Test samples  : {test_gen.samples}")

print(f"\nClasses : {train_gen.class_indices}")

# SAVE CLASS INDICES
os.makedirs("backend/model", exist_ok=True)

with open("backend/model/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

print("\n[2/6] Building model...")

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

backbone = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
)

# FREEZE BACKBONE INITIALLY
backbone.trainable = False

x = backbone.output

# SIMPLE HEAD (BETTER FOR TRANSFER LEARNING)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.summary()

# ─────────────────────────────────────────────────────────────
# PHASE 1 - TRAIN HEAD
# ─────────────────────────────────────────────────────────────

print("\n[3/6] Phase 1 - Training head...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    ),
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_phase1,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────
# PHASE 2 - FINE TUNING
# ─────────────────────────────────────────────────────────────

print("\n[4/6] Phase 2 - Fine tuning...")

backbone.trainable = True

# ONLY TRAIN TOP 20 LAYERS
for layer in backbone.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-8
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_phase2,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

print("\n[5/6] Evaluating model...")

test_gen.reset()

y_pred_probs = model.predict(test_gen)

y_pred = np.argmax(y_pred_probs, axis=1)

y_true = test_gen.classes

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASSES
    )
)

# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

print("\n[6/6] Saving plots...")

os.makedirs("notebooks/plots", exist_ok=True)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "notebooks/plots/confusion_matrix.png",
    dpi=150
)

plt.close()

print(f"\nModel saved to: {MODEL_OUT}")
print("Training complete.")