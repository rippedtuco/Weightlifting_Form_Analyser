import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

# Read in the data from the CSV file
bicep_misc_angles_data= pd.read_csv('/content/drive/MyDrive/bicep_misc_angles.csv')

# Extract the correct body angles and labels from the data
elbow_angles= bicep_misc_angles_data.iloc[:, 0].values.reshape(-1, 1)
back_angles = bicep_misc_angles_data.iloc[:, 1].values.reshape(-1, 1)



# Combine the two columns into a single array
angles_all = np.concatenate((elbow_angles, back_angles), axis=1)

# Set all labels to 1 since the angles are correct
labels_all = np.ones(len(angles_all))


# Combine the angles and labels into a single DataFrame
bicep_misc_angles_data = pd.DataFrame(np.concatenate((angles_all, labels_all.reshape(-1, 1)), axis=1), columns=['elbow_angle', 'back_angle', 'label'])

# Split the data into training, validation, and test sets
train_angles_misc, test_angles_misc, train_labels_misc, test_labels_misc = train_test_split(angles_all, labels_all, test_size=0.2, random_state=42)

train_angles_misc, val_angles_misc, train_labels_misc, val_labels_misc = train_test_split(train_angles_misc, train_labels_misc, test_size=0.2, random_state=42)