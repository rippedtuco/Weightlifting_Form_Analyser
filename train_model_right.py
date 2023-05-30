from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.regularizers import l2

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), input_dim=2))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
history =model.fit(train_angles_right, train_labels_right, validation_data=(val_angles_right, val_labels_right), epochs=20, batch_size=10)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_angles_right, test_labels_right)
print('Test accuracy:', accuracy)

model.save('bicep_right.h5')