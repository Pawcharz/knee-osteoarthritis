import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.other import get_confusion_matrix_display

class Trainer:
  
  def __init__(self, model, classes, train_loader, val_loader, criterion, optimizer, device, lr_scheduler=None, reg_type = None, reg_lambda=0, tensorboard_logger=None):
    
    self.model = model
    self.classes = classes
    
    self.train_loader = train_loader
    self.val_loader = val_loader
    
    self.criterion = criterion
    self.optimizer = optimizer
    
    self.device = device
    
    self.reg_type = reg_type
    self.reg_lambda = reg_lambda
    
    self.lr_scheduler = lr_scheduler
    
    self.logger = tensorboard_logger
    
    self.epochCounter = 0
    
  def get_lr(self):
    for param_group in self.optimizer.param_groups:
      return param_group['lr']
  
  def evaluate(self, loader):
    self.model.eval()
    
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
      for inputs, labels in loader:
        labels = labels.to(self.device)
        
        outputs = self.model(inputs).to(self.device)
        _, predicted = torch.max(outputs, 1)
        
        loss = self.criterion(outputs, labels)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    report = classification_report(true_labels, predicted_labels, zero_division=np.nan)
    cm = confusion_matrix(true_labels, predicted_labels)
  
    return accuracy, loss, report, cm

  # Logged training loss does not include regularization for it to be more readable during analysis / comparing with valdiation
  def train_many(self, epochs_nr):    
    for epoch in range(0, epochs_nr):  # loop over the dataset multiple times
      correct = 0
      total = 0
      predicted_labels = []
      true_labels = []
  
      for i, data in enumerate(self.train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        labels = labels.to(self.device)
        
        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # forward -> calculating loss
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)
        tLoss = self.criterion(outputs, labels)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        tAccuracy = 100 * correct / total
    
        # Apply L1 regularization
        if self.reg_type == 'L1':
          l1_norm = sum(p.abs().sum() for p in self.model.parameters())
          tLoss += self.reg_lambda * l1_norm
            
        # Apply L2 regularization
        elif self.reg_type == 'L2':
          l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
          tLoss += self.reg_lambda * l2_norm
            
        # Backward propagation
        tLoss.backward()
        self.optimizer.step()
        
        # Changing outputs (logits) to labels
        # outputs_clear = outputs.max(1).indices
        
        # epoch_correct += (outputs_clear == labels).float().sum()
        # epoch_samples += len(outputs)
        # epoch_batches +=1
        
        # running_loss += loss.item()
    
      # tAccuracy = epoch_correct / epoch_samples * 100
      # tLoss = running_loss / epoch_batches
      
      # Validation + Training report and confusion matrix
      vAccuracy, vLoss, vReport, vCm = self.evaluate(self.val_loader)
      tReport = classification_report(true_labels, predicted_labels, zero_division=np.nan)
      tCm = confusion_matrix(true_labels, predicted_labels)
      
      learning_rate = self.get_lr()
      
      if self.logger != None:
        self.logger.add_text("REGULARIZATION_TYPE", self.reg_type, self.epochCounter)
        self.logger.add_scalar("REGULARIZATION_LAMBDA", self.reg_lambda, self.epochCounter)
        self.logger.add_scalar("learning_rate", learning_rate, self.epochCounter)
        
        self.logger.add_scalars("Accuracy", {
          "training": tAccuracy,
          "validation": vAccuracy,
        }, self.epochCounter)
        
        self.logger.add_scalars("Loss", {
          "training": tLoss,
          "validation": vLoss,
        }, self.epochCounter)
        
        if self.epochCounter % 5 == 0:
          # print(cos)
          self.logger.add_text("Classification Report/train", tReport, global_step=self.epochCounter)
          self.logger.add_text("Classification Report/validation", vReport, global_step=self.epochCounter)
          self.logger.add_figure("Training Confusion matrix/train", get_confusion_matrix_display(tCm, self.classes, "Training", self.epochCounter), global_step=self.epochCounter)
          self.logger.add_figure("Validation Confusion matrix/validation", get_confusion_matrix_display(vCm, self.classes, "Validation", self.epochCounter), global_step=self.epochCounter)
          
      print(f'Epoch {self.epochCounter}: Training: accuracy: {tAccuracy:.3f}%, loss: {tLoss:.3f}; Validation: accuracy: {vAccuracy:.3f}%, loss: {vLoss:.3f}, lr: {learning_rate:.5f}')
      
      self.epochCounter += 1
      
      if self.lr_scheduler != None:
        self.lr_scheduler.step()
      
      # print("lr= " + str(learning_rate))
    print('Finished Training')