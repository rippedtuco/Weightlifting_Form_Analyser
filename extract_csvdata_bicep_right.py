import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

# Read in the data from the CSV file
right_bicep_angles_data_correct = pd.read_csv('/content/drive/MyDrive/bicep_right_angle.csv')
right_bicep_angles_data_incorrect = pd.read_csv('/content/drive/MyDrive/right_wrong.csv')

# Extract the correct body angles and labels from the data
elbow_angles_correct = right_bicep_angles_data_correct.iloc[:, 0].values.reshape(-1, 1)
back_angles_correct = right_bicep_angles_data_correct.iloc[:, 1].values.reshape(-1, 1)

# Extract the incorrect body angles and labels from the data
elbow_angles_incorrect = right_bicep_angles_data_incorrect.iloc[:, 0].values.reshape(-1, 1)
back_angles_incorrect = right_bicep_angles_data_incorrect.iloc[:, 1].values.reshape(-1, 1)

# Combine the two columns into a single array
angles_correct = np.concatenate((elbow_angles_correct, back_angles_correct), axis=1)
angles_incorrect = np.concatenate((elbow_angles_incorrect, back_angles_incorrect), axis=1)

# Set all labels to 1 since the angles are correct
labels_correct = np.ones(len(angles_correct))

# Set labels to 0 for incorrect angles
labels_incorrect = np.zeros(len(angles_incorrect))

# Concatenate the correct and incorrect body angles into a single array
angles_all = np.concatenate((angles_correct, angles_incorrect), axis=0)

# Create labels for all angles
labels_all = np.concatenate((labels_correct,labels_incorrect), axis=0)

# Combine the angles and labels into a single DataFrame
right_bicep_angles_data = pd.DataFrame(np.concatenate((angles_all, labels_all.reshape(-1, 1)), axis=1), columns=['elbow_angle', 'back_angle', 'label'])


# Split the data into training, validation, and test sets
train_angles_right, test_angles_right, train_labels_right, test_labels_right = train_test_split(angles_all, labels_all, test_size=0.2, random_state=42)

train_angles_right, val_angles_right, train_labels_right, val_labels_right = train_test_split(train_angles_right, train_labels_right, test_size=0.2, random_state=42)