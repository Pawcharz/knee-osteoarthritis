from collections import Counter
from torch.utils.data import DataLoader
import numpy as np
import torch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_classes_frequencies(dataset):
  freq_table = dict(Counter(dataset.labels))
  return freq_table

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
