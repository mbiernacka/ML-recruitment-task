from models import *

# For the NN model training was tested only on 100 rows to make it quick
df.sample(frac=1)
print(X_train)
print(y_train)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set hyperparameters to choose the best combination from
hyperparameters = {
    'learning_rate': [0.01, 0.001],
    'hidden_layers': [(64,), (128,), (256,), (64, 64)],
    'dropout': [0.1, 0.2, 0.3],
    'epochs': [10, 20, 30],
    'batch_size': [32, 64]
}


# Create NN with default hyperparameters
def create_neural_network(learning_rate=0.001, hidden_layers=(64,), dropout=0.1):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layers[0], activation='relu', input_dim=54),
        tf.keras.layers.Dropout(dropout)
    ])
    for hidden_layer_size in hidden_layers[1:]:
        model.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Find the best hyperparameters using GridSearchCV
model = KerasClassifier(build_fn=create_neural_network, verbose=1)
grid_search = GridSearchCV(model, hyperparameters, cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Print the best hyperparameters and accuracy
best_params = grid_result.best_params_
best_score = grid_result.best_score_
print(f'Best Hyperparameters: {best_params}')
print(f'Best Accuracy: {best_score:.2f}')


# Plot the training curves for the best hyperparameters

history = grid_result.best_estimator_.model.history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()