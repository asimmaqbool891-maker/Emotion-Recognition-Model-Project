# ============================================================
# CELL 1: IMPORT LIBRARIES
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

print("="*60)
print("🎭 EMOTION RECOGNITION FROM SPEECH - FINAL FIXED")
print("="*60)

# ============================================================
# CELL 2: CREATE SYNTHETIC DATA (FIXED SIZE)
# ============================================================

print("\n📊 Creating dataset...")

np.random.seed(42)
n_samples = 10000

EMOTION_INFO = {
    'neutral': {'emoji': '😐', 'color': '#95a5a6'},
    'calm': {'emoji': '😌', 'color': '#3498db'},
    'happy': {'emoji': '😊', 'color': '#f1c40f'},
    'sad': {'emoji': '😢', 'color': '#34495e'},
    'angry': {'emoji': '😠', 'color': '#e74c3c'},
    'fearful': {'emoji': '😨', 'color': '#9b59b6'},
    'disgust': {'emoji': '🤢', 'color': '#2ecc71'},
    'surprised': {'emoji': '😲', 'color': '#e67e22'}
}

emotions = list(EMOTION_INFO.keys())

# FIXED: All arrays same size (120 features)
X_list = []
y_list = []

for emotion in emotions:
    n = n_samples // len(emotions)

    # FIXED: Create base pattern with EXACTLY 120 elements
    if emotion == 'happy':
        # 120 elements: 8*5 + 40 + 40 + 32 = 40 + 40 + 40 + 0 = 120
        base = np.array([1.5, 1.3, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3] * 5 +  # 40 elements
                       [0.9] * 40 +  # 40 elements  
                       [0.7] * 30 +  # 30 elements
                       [0.5] * 10)   # 10 elements

    elif emotion == 'sad':
        base = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.1] * 5 +
                       [-0.7] * 40 +
                       [-0.5] * 30 +
                       [-0.3] * 10)

    elif emotion == 'angry':
        base = np.array([0.8, 1.2, 0.6, 1.5, 0.4, 1.0, 0.2, 0.9] * 5 +
                       [0.85] * 40 +
                       [0.75] * 30 +
                       [0.65] * 10)

    elif emotion == 'fearful':
        base = np.array([-0.3, 0.8, -0.5, 1.2, -0.2, 0.9, -0.4, 0.6] * 5 +
                       [0.3] * 40 +
                       [0.2] * 30 +
                       [0.1] * 10)

    elif emotion == 'disgust':
        base = np.array([-0.8, -0.4, -1.0, -0.2, -0.6, -0.1, -0.9, 0.0] * 5 +
                       [-0.5] * 40 +
                       [-0.4] * 30 +
                       [-0.3] * 10)

    elif emotion == 'surprised':
        base = np.array([0.5, 1.5, 0.3, 1.8, 0.2, 1.2, 0.4, 1.0] * 5 +
                       [0.9] * 40 +
                       [0.8] * 30 +
                       [0.7] * 10)

    elif emotion == 'calm':
        base = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.2] * 5 +
                       [0.18] * 40 +
                       [0.16] * 30 +
                       [0.14] * 10)

    else:  # neutral
        base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 5 +
                       [0.0] * 40 +
                       [0.0] * 30 +
                       [0.0] * 10)

    # Verify size
    assert len(base) == 120, f"Base size is {len(base)}, expected 120"

    # Create samples with small noise
    features = np.random.randn(n, 120) * 0.15 + base

    X_list.append(features)
    y_list.extend([emotion] * n)

X = np.vstack(X_list)
y = np.array(y_list)

print(f"✅ Dataset created: {len(X)} samples, {X.shape[1]} features")

# ============================================================
# CELL 3: VISUALIZE
# ============================================================

print("\n📈 Visualizing...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Distribution
emotion_counts = pd.Series(y).value_counts()
colors = [EMOTION_INFO[e]['color'] for e in emotion_counts.index]
axes[0,0].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', colors=colors)
axes[0,0].set_title('Emotion Distribution')

# Feature 0 by emotion
for emotion in emotions:
    mask = y == emotion
    axes[0,1].hist(X[mask, 0], alpha=0.5, label=emotion, color=EMOTION_INFO[emotion]['color'])
axes[0,1].set_title('Feature Pattern')
axes[0,1].legend()

# Correlation
sns.heatmap(pd.DataFrame(X[:, :10]).corr(), annot=True, fmt='.1f', ax=axes[1,0])
axes[1,0].set_title('Feature Correlation')

# Box plot
pd.DataFrame({f'F{i}': X[:, i] for i in range(5)}).boxplot(ax=axes[1,1])
axes[1,1].set_title('Feature Distribution')

plt.tight_layout()
plt.savefig('/kaggle/working/emotion_analysis.png', dpi=150)
plt.show()

# ============================================================
# CELL 4: PREPARE DATA
# ============================================================

print("\n⚙️ Preparing data...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

print(f"✅ Data ready: Train={len(X_train)}, Test={len(X_test)}")

# ============================================================
# CELL 5: BUILD MODEL
# ============================================================

print("\n🧠 Building model...")

model = models.Sequential([
    layers.Input(shape=(X_train_cnn.shape[1], 1)),

    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(32),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CELL 6: TRAIN
# ============================================================

print("\n🚀 Training...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train_cnn, y_train_cat,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# CELL 7: EVALUATE
# ============================================================

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)

test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_classes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
ax1.set_title('Confusion Matrix')

ax2.plot(history.history['accuracy'], label='Train', color='green')
ax2.plot(history.history['val_accuracy'], label='Val', color='blue')
ax2.axhline(y=test_acc, color='red', linestyle='--', label=f'Test: {test_acc:.3f}')
ax2.set_title('Training History')
ax2.legend()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/kaggle/working/training_results.png', dpi=150)
plt.show()

print("\n📊 Per-Class Accuracy:")
for i, emotion in enumerate(le.classes_):
    acc = cm[i, i] / cm[i, :].sum()
    print(f"   {EMOTION_INFO[emotion]['emoji']} {emotion:12s}: {acc:.2%}")

# ============================================================
# CELL 8: SAVE
# ============================================================

print("\n💾 Saving...")

model.save('/kaggle/working/emotion_model.h5')

import pickle
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('/kaggle/working/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import json
with open('/kaggle/working/results.json', 'w') as f:
    json.dump({
        'accuracy': float(test_acc),
        'emotions': le.classes_.tolist()
    }, f)

print("✅ All saved!")

print("\n" + "="*60)
print("🎉 TASK 2 COMPLETE!")
print("="*60)
print(f"\n🏆 Final Accuracy: {test_acc*100:.2f}%")
