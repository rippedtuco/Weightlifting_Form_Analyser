from keras.models import Model
from keras.layers import concatenate

# Load the saved models
bicep_right = load_model('bicep_right.h5')
bicep_left = load_model('bicep_left.h5')
bicep_misc = load_model('bicep_misc.h5')

# Get the output layers of each model
output_right = bicep_right.layers[-1].output
output_left = bicep_left.layers[-1].output
output_misc = bicep_misc.layers[-1].output

# Concatenate the output layers
merged = concatenate([output_right, output_left, output_misc])

# Define a new model with the concatenated output
bicep_model = Model(inputs=[bicep_right.input, bicep_left.input, bicep_misc.input], outputs=merged)
bicep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])