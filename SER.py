import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import joblib  # Import joblib

# 1. Preprocess Data

# Load the data
dataset_path = '.csv' # a csv with the preprocessed data
data = pd.read_csv(dataset_path)

# Extract features
feature_cols = [str(i) for i in range(171)]
X = data[feature_cols].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Need to reshape data for Conv1D layers
X = np.expand_dims(X, axis=-1)

# Extract and encode labels
y = data['labels'].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 2. Define the Model

L = tf.keras.layers
model = tf.keras.Sequential([
    L.Conv1D(128, kernel_size=5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=2, strides=2, padding='same'),

    L.Conv1D(256, kernel_size=5, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=2, strides=2, padding='same'),

    L.Flatten(),
    L.Dense(128, activation='relu'),
    L.BatchNormalization(),
    L.Dense(y_categorical.shape[1], activation='softmax')  # Number of unique emotion classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 4. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 5. Generate Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# 6. Save the Model
# Create a dictionary to store model architecture and weights
model_dict = {
    'model_config': model.get_config(),
    'model_weights': model.get_weights()
}

# Save to .pkl file
model_path = ''#provide here the path to the model
joblib.dump(model_dict, model_path)

