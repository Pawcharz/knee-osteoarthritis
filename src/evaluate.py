import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.other import getConfusionMatrixDisplay
    
def evaluate_augmented_model(model, criterion, loader, device):
  correct = 0
  total = 0
  model.eval()
  predicted_labels = []
  true_labels = []

  with torch.no_grad():
    for inputs, edges, labels in loader:
      inputs, edges, labels = inputs.to(device), edges.to(device), labels.to(device)
      outputs = model(inputs, edges)
      _, predicted = torch.max(outputs, 1)
      loss = criterion(outputs, labels)
      
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      predicted_labels.extend(predicted.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())

  accuracy = 100 * correct / total
  report = classification_report(true_labels, predicted_labels, zero_division=np.nan)
  cm = confusion_matrix(true_labels, predicted_labels)
  
  return accuracy, loss, report, cm