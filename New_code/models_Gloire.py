import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils import resample


# SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fourth convolutional layer to match output dimensions
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply final conv layer
        x = torch.sigmoid(self.conv4(x))

        return x


# Training function
def train_simple_cnn_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, mask in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Compute element-wise loss
            masked_loss = loss * mask  # Apply mask to loss
            loss = masked_loss.mean()  # Compute mean of masked loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# Improved SimpleCNN model
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.conv4(x))
        return x


# Data augmentation function
def augment_data(inputs, labels, mask):
    # Flip horizontally with 50% probability
    if random.random() > 0.5:
        inputs = torch.flip(inputs, [3])
        labels = torch.flip(labels, [3])
        mask = torch.flip(mask, [3])

    # Flip vertically with 50% probability
    if random.random() > 0.5:
        inputs = torch.flip(inputs, [2])
        labels = torch.flip(labels, [2])
        mask = torch.flip(mask, [2])

    return inputs, labels, mask


# Training function
def train_improved_cnn_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, mask in train_loader:
            inputs, labels, mask = augment_data(inputs, labels, mask)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            masked_loss = loss * mask
            loss = masked_loss.sum() / mask.sum()  # Weighted loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_loss)


# Evaluation function
def evaluate_cnn_model(model, val_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels, mask in val_loader:
            outputs = model(inputs).squeeze()

            # Flatten the outputs, labels, and mask
            outputs_flat = outputs.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            mask_flat = mask.view(-1).cpu().numpy()

            # Apply the mask
            valid_indices = mask_flat > 0
            all_labels.extend(labels_flat[valid_indices])
            all_preds.extend(outputs_flat[valid_indices])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    predicted_classes = (all_preds > 0.5).astype(int)

    # Calculate metrics
    accuracy = np.mean(all_labels == predicted_classes)
    balanced_acc = balanced_accuracy_score(all_labels, predicted_classes)
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


class BootstrappedSoftVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10, xg_boost=False):
        self.xg_boost = xg_boost
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            # Create a bootstrap sample
            X_bootstrap, y_bootstrap = resample(X, y)

            # Create and fit a new estimator
            estimator = clone(self.base_estimator)
            class_counts = Counter(y_bootstrap)
            scale_pos_weight = class_counts[0] / class_counts[1]

            # Calculate the class_weight hyperparameter for SVM
            class_weight_options = {0: 1, 1: scale_pos_weight}
            if self.xg_boost:
                estimator.set_params(scale_pos_weight=scale_pos_weight)
            else:
                estimator.set_params(class_weight=class_weight_options)
            estimator.fit(X_bootstrap, y_bootstrap)
            # self.classes_ = estimator.classes_
            if not np.array_equal(estimator.classes_, self.classes_):
                raise ValueError("All estimators must have the same class order")

            self.estimators_.append(estimator)
        return self

    def predict_proba(self, X):
        # Collect predictions from all estimators
        all_proba = np.array([est.predict_proba(X) for est in self.estimators_])

        # Average the probabilities (soft voting)
        avg_proba = np.mean(all_proba, axis=0)

        return avg_proba

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return self.classes_[np.argmax(avg_proba, axis=1)]
