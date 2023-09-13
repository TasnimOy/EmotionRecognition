import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model from the .pkl file
model_path = '.pkl'#Provide here a path to the .pkl model
loaded_model_dict = joblib.load(model_path)
loaded_model = tf.keras.models.Sequential.from_config(loaded_model_dict['model_config'])
loaded_model.set_weights(loaded_model_dict['model_weights'])
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the new preprocessed data
new_data_path = '.csv'#Provide here code with the preprocessed data
new_data = pd.read_csv(new_data_path)

# Exclude the 'file_path' column for prediction
X_new = new_data.drop(columns=['file_path']).values

# Predict emotions on the new dataset
y_new_pred = loaded_model.predict(X_new)

# If you want to get the predicted class labels instead of probabilities:
y_new_pred_classes = np.argmax(y_new_pred, axis=1)

# Mapping from emotion labels to emotion names
emotion_dict = {0:'neutral', 1: 'happy', 2: 'sad', 3: 'Pleasent_Surprise', 4: 'fearful', 5: 'surprise',
                6: 'angry', 7: 'disgust', 8: 'calm', 9: 'excited', 10: 'frustrated', 11: 'boredom'}

# Convert numerical labels to emotion names
y_new_pred_emotions = [emotion_dict[label] for label in y_new_pred_classes]

# Print predicted class labels
print(y_new_pred_emotions)

# Prepare data for the evaluation CSV
evaluation_data = {
    "file_path": new_data['file_path'].values,
    "emotion": y_new_pred_emotions
}

evaluation_df = pd.DataFrame(evaluation_data)
evaluation_df.to_csv('.csv', index=False)# provide a path where u want to store the final csv

# Plotting distribution of predicted emotions
plt.figure(figsize=(15,7))
evaluation_df['emotion'].value_counts().plot(kind='bar')
plt.title("Distribution of Predicted Emotions")
plt.ylabel("Number of Predictions")
plt.xlabel("Emotion")
plt.savefig(".png") #Provide a path where u want to save the evaluation
