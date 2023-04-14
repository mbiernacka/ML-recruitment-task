from main import *

# Train a decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
dtc_accuracy = dtc.score(X_test, y_test)
print(f'Decision Tree Classifier Accuracy: {dtc_accuracy:.2f}')


# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_accuracy = rfc.score(X_test, y_test)
print(f'Random Forest Classifier Accuracy: {rfc_accuracy:.2f}')

