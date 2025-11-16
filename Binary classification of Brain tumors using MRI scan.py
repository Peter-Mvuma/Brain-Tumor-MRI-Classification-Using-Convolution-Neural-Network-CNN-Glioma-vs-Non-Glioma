# Brain Tumor classification - Binary Glioma vs Non-Glioma CNN with Spark 

# Importing all the necessary libaries for the work
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import IntegerType
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Configuring the core Model and data pipeline parameters.
BASE_DIR = "/home/sat3812/brain_tumor_images"  
IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 10
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# Initializing the Spark session for distributed processing.
print("=== Starting Spark ===")
spark = (
    SparkSession.builder
    .appName("Binary_Glioma_CNN_Spark_32x32")
    .getOrCreate()
)

print(f"Base directory: {BASE_DIR}")

# Assemble image paths and labels into Spark DataFrame.
rows = []
for label_str in ["1", "2", "3"]:
    label_dir = os.path.join(BASE_DIR, label_str)
    if not os.path.isdir(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            rows.append((os.path.join(label_dir, fname), label_str))

if not rows:
    print("ERROR: No images found – check BASE_DIR.")
    spark.stop()
    exit(1)

schema = ["path", "label_str"]
df = spark.createDataFrame(rows, schema=schema)

# Map original class labels to integer indices that is class 1=meningioma (0), 2=glioma (1), 3=pituitary (2)
df = df.withColumn(
    "label_index",
    (col("label_str").cast(IntegerType()) - lit(1)).cast(IntegerType())
)

# Generate binary label - glioma=1, others=0.
 glioma (original class 2 → index 1) = 1; others = 0
df = df.withColumn(
    "binary_label",
    (col("label_index") == 1).cast(IntegerType())
)

print("Sample of images with binary labels:")
df.select("path", "label_str", "label_index", "binary_label").show(5, truncate=False)

total_count = df.count()
print(f"Total images: {total_count}")

# Split the data into train, validation, and test sets
test_frac = TEST_FRACTION
val_frac = VAL_FRACTION
train_frac = 1.0 - test_frac - val_frac

train_df, val_df, test_df = df.randomSplit(
    [train_frac, val_frac, test_frac],
    seed=SEED
)

train_count = train_df.count()
val_count = val_df.count()
test_count = test_df.count()
print(f"Train count: {train_count}")
print(f"Val count:   {val_count}")
print(f"Test count:  {test_count}")

# Load image data into NumPy arrays (32x32, grayscale)
def load_split_to_numpy(rows, img_size=IMG_SIZE):
    """
    rows: list of Row(path=..., binary_label=...)
    returns: X (N, img_size, img_size, 1), y (N,)
    """
    n = len(rows)
    X = np.zeros((n, img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)

    for i, r in enumerate(rows):
        path = r["path"]
        label = r["binary_label"]
        try:
            img = Image.open(path).convert("L")  # grayscale
            img = img.resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32) / 255.0  # (img_size, img_size)
            # Add channel dimension
            X[i, :, :, 0] = arr
            y[i] = label
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")

    return X, y

print("=== Loading image data into NumPy arrays (32x32, grayscale) ===")
train_rows = train_df.select("path", "binary_label").collect()
val_rows = val_df.select("path", "binary_label").collect()
test_rows = test_df.select("path", "binary_label").collect()

X_train, y_train = load_split_to_numpy(train_rows, IMG_SIZE)
X_val, y_val = load_split_to_numpy(val_rows, IMG_SIZE)
X_test, y_test = load_split_to_numpy(test_rows, IMG_SIZE)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape: {y_test.shape}")

# Generate class weights for training
classes = np.unique(y_train.astype(int))
class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights_values)}
print("Class weights (0=non-glioma, 1=glioma):", class_weight_dict)

# Build binary classification CNN
print("=== Building binary CNN model (32x32) ===")
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Fit model to training data
print("=== Training binary CNN model ===")
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    verbose=2
)

# Test set evaluation
print("=== Evaluating on test set ===")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Performance metrics and reports
print("=== Computing confusion matrix and classification report (binary) ===")
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int).flatten()
y_true = y_test.astype(int)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["non-glioma (0)", "glioma (1)"]
))

# Save training curves
history_dict = history.history

plt.figure()
plt.plot(history_dict["loss"], label="train_loss")
plt.plot(history_dict["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Binary CNN – Loss curve")
plt.tight_layout()
plt.savefig("binary_loss_curve.png")

plt.figure()
plt.plot(history_dict["accuracy"], label="train_acc")
plt.plot(history_dict["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Binary CNN – Accuracy curve")
plt.tight_layout()
plt.savefig("binary_accuracy_curve.png")

print("Saved binary_loss_curve.png and binary_accuracy_curve.png in current directory.")

spark.stop()
print("=== Done ===")
