import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from GeneticNN import *


pop_size = int(input("Insira o tamanho da população: "))
X = np.load('X32.npy')
dims = (32**2)*3
y = np.load('y.npy')


# Redimensionamento e undersampling
X = X.reshape(len(X), dims)
healthy = pd.DataFrame(X[y==0])
healthy['label'] = False
sick = pd.DataFrame(X[y!=0][:len(healthy)])
sick['label'] =  True

data = pd.concat([healthy, sick])
columns = []
for col in data.columns:
    if col != 'label':
        columns.append(col)
X = data[columns]
y = data.label
train_images, test_images, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Normalização
train_images, test_images = train_images / 255.0, test_images / 255.0

# Rodando modelo geneticamente otimizado
clf = GeneticOptimizedNNModel(pop_size)

clf.run(train_images, test_images, train_labels, test_labels, 0.60, 0.05)
