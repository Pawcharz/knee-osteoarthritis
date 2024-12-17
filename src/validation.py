import torch

def validate(model, val_loader, criterion, device):
  with torch.no_grad():
    epoch_correct = 0
    epoch_samples = 0
    epoch_batches = 0
    running_loss = 0.0
    
    for i, data in enumerate(val_loader):
      images, edges, labels = data
      images = images.to(device)
      edges = edges.to(device)
      labels = labels.to(device)
      
      # forward + backward + optimize
      outputs = model(images, edges)
      loss = criterion(outputs, labels)
      
      # Changing outputs (logits) to labels
      outputs_clear = outputs.max(1).indices
      
      epoch_correct += (outputs_clear == labels).float().sum()
      epoch_samples += len(outputs)
      
      epoch_batches += 1
      
      running_loss += loss.item()
    
    # print(epoch_correct, epoch_samples)
    accuracy = epoch_correct / epoch_samples * 100
    loss = running_loss / epoch_batches
    
    return accuracy, loss
    