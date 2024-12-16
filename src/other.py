from collections import Counter

def getClassesFrequency(dataset):
  freq_table = dict(Counter(dataset.labels))
  return freq_table