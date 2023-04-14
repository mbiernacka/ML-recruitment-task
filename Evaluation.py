from NN import *
from models import *

# Define the models to evaluate
models = [('Decision Tree', dtc),
          ('Random Forest', rfc),
          ('Neural Network', grid_result.best_estimator_)]

# Evaluate each model on the test set and store the results
results = {}
for name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1,
                     'Confusion Matrix': cm}

# Print the results
for name, metrics in results.items():
    print(f'{name}:')
    for metric_name, metric_value in metrics.items():
        print(metric_name, metric_value)
    print()

# Plot the accuracy scores of each model
accuracy_scores = [results[name]['Accuracy'] for name, _ in models]
plt.bar([name for name, _ in models], accuracy_scores)
plt.title('Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()
