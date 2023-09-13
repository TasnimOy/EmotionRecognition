import pandas as pd
import matplotlib.pyplot as plt

# Paths to the CSV files
paths = [
         ]  #Provide here the .csv files with the evaluation.

# Define emotion labels
emotion_labels = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'Pleasent_Surprise', 4: 'fearful', 5: 'surprise',
                  6: 'angry', 7: 'disgust', 8: 'calm', 9: 'excited', 10: 'frustrated', 11: 'boredom'}

# Initialize a DataFrame to store the counts
all_counts = pd.DataFrame(columns=emotion_labels.values())

for idx, path in enumerate(paths):
    # Read the CSV file
    data = pd.read_csv(path)

    # Replace emotion numbers with emotion labels
    data.replace({"emotion": emotion_labels}, inplace=True)

    # Count the number of each emotion
    emotion_counts = data['emotion'].value_counts()

    # Append counts to the all_counts DataFrame
    all_counts.loc[idx] = emotion_counts

# Create a stacked bar chart
all_counts.transpose().plot(kind='bar', stacked=True, figsize=(15, 7))

# Configure the chart
plt.title("Distribution of Predicted Emotions from Multiple Sessions")
plt.ylabel("Number of Predictions")
plt.xlabel("Emotion")
plt.tight_layout()

# Extract session number from path and create the legend accordingly
session_numbers = [f'p{path.split("Session_")[1].split("/")[0].split(".")[0]}' for path in paths]
plt.legend(session_numbers)

# Save the figure
plt.savefig(".png")# provide the path where u want to save the evaluation

# Show the plot
plt.show()
