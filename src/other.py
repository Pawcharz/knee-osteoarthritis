from collections import Counter
from torch.utils.data import DataLoader
import numpy as np
import torch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def getClassesFrequency(dataset):
  freq_table = dict(Counter(dataset.labels))
  return freq_table

def getWeightedSampler(dataset):
  freq_table = getClassesFrequency(dataset)
  least_class_frequency = min(freq_table.values())

  normalized_freq = dict(freq_table)
  for i, val in enumerate(normalized_freq.values()):
    normalized_freq[i] = val / len(dataset.labels)
    
  print(normalized_freq, least_class_frequency)
  
  weights = np.zeros(len(dataset.labels))
  for i, weight in enumerate(weights):
    label = dataset.labels[i]
    weights[i] = 1 / freq_table[label]
      
  # print(weights)
  samples_weight = torch.from_numpy(weights)
  samples_weigth = samples_weight.double()
  sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
  
  return sampler

def getWeightedDataLoader(dataset, batch_size):
  sampler = getWeightedSampler(dataset)
  
  dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
  
  return dataloader

def get_confusion_matrix_display(cm, classes, evType, epoch=None):
  # Build confusion matrix
  df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
  plt.figure(figsize=(4, 3))    
  heatmap = sns.heatmap(df_cm, cmap='crest', annot=True)
  
  title = evType
    
  if epoch != None:
    title += f", epoch: {epoch}"
      
  heatmap.set(xlabel='true label', ylabel='predicted label', title=title)
  return heatmap.get_figure()