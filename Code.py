import pandas as pd
import sklearn
import numpy as np
import os
import soundfile as sf
import librosa.feature

# Load Dataset
csv_path = "D:/edge/Deception-main/Deception-main/CBU0521DD_stories_attributes.csv"
audio_folder = "D:/edge/Deception-main/Deception-main/CBU0521DD_stories/"
data = pd.read_csv(csv_path)

# Display dataset information
data.info()
print(data.head())

# Rename columns for consistency
data.columns = ["filename", "Language", "Story_type"]

# Encode Language Feature
language_encoder = sklearn.preprocessing.LabelEncoder()
languages = language_encoder.fit_transform(data['Language'])

# Extract Features from Audio Files
features = []
labels = []

for index, row in data.iterrows():
    file_path = os.path.join(audio_folder, row['filename'])
    label = row['Story_type']
    try:
        # Load audio file using soundfile for basic statistics
        signal, sr = sf.read(file_path)
        mean = np.mean(signal)
        std = np.std(signal)

        # Load audio again using librosa for more detailed features
        y, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs).T, axis=0)
        mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        # Combine simple and complex features
        feature = np.hstack([mean, std, mfccs, mfcc_delta, mfcc_delta2, centroid, zcr, rms, contrast])
        feature = np.hstack([feature, languages[index]])

        features.append(feature)
        labels.append(label)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode Labels
encoder = sklearn.preprocessing.LabelEncoder()
labels = encoder.fit_transform(labels)

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the features
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Checking the distribution of classes in training and testing sets
print(f"Training Set: True Stories: {sum(y_train == 1)}, Deceptive Stories: {sum(y_train == 0)}")
print(f"Testing Set: True Stories: {sum(y_test == 1)}, Deceptive Stories: {sum(y_test == 0)}")

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize class distribution
sns.countplot(x=labels, order=[1, 0])
plt.xlabel("Story Type")
plt.ylabel("Count")
plt.title("Class Distribution (True vs. Deceptive)")
plt.xticks(ticks=[0, 1], labels=["Deceptive", "True"])
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=15, min_samples_split=4, min_samples_leaf=4, random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Check training accuracy
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

#  Evaluate Model
y_pred = clf.predict(X_test)

# Calculate testing accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Generate Classification Report
print("Classification Report (Testing Set):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Testing Set):")
print(cm)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap (Testing Set)')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
ensemble_clf = RandomForestClassifier(n_estimators=7, max_depth=15, min_samples_split=4, min_samples_leaf=4, random_state=42)
ensemble_clf.fit(X_train, y_train)

# Check training accuracy
y_train_pred_ensemble = ensemble_clf.predict(X_train)
train_accuracy_ensemble = accuracy_score(y_train, y_train_pred_ensemble)
print(f"Training Accuracy: {train_accuracy_ensemble * 100:.2f}%")

# Make predictions
y_pred_ensemble = ensemble_clf.predict(X_test)

# Evaluation
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Testing Accuracy: {accuracy_ensemble * 100:.2f}%")
print("Classification Report (Testing Set):")
print(classification_report(y_test, y_pred_ensemble))

# Confusion Matrix
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
print("Confusion Matrix (Testing Set):")
print(cm_ensemble)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Greens', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap (Testing Set)')
plt.show()