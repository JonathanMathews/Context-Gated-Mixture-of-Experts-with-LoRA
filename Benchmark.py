import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import BeitForImageClassification, BeitImageProcessor

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import datasets
import gc


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

BS = 64
FIRE_PATH = './fire_dataset'
GAME_PATH = './game_dataset'
RESULTS_PATH = './Results/Benchmark/'
BEIT_MODEL = 'microsoft/beit-large-patch16-224'

fire = datasets.DatasetDict.load_from_disk(FIRE_PATH)
game = datasets.DatasetDict.load_from_disk(GAME_PATH)

print('Data Loaded!')

imageProcessor = BeitImageProcessor.from_pretrained(BEIT_MODEL)

def fullTransforms(examples):

    examples['pixel_values'] = imageProcessor([image.convert('RGB') for image in examples['image']],
                                              return_tensors = 'pt')['pixel_values']

    examples['image'] = examples['pixel_values']
    del examples['pixel_values']

    return examples

fire = fire.with_transform(fullTransforms)
game = game.with_transform(fullTransforms)

print('Preprocessing Finished!')

fireTest = DataLoader(fire['test'], batch_size = BS, shuffle = True)
gameTest = DataLoader(game['test'], batch_size = BS, shuffle = True)

print('Data Loaders Created!')

model = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                   num_labels = 4,
                                                   ignore_mismatched_sizes = True)
model.to(device)
model.eval()

print('BEiT Model Initialized!')

# Fire Risk Classification

allPredictions = []
allTargets = []

for _, data in enumerate(fireTest, 0):

    inputImages = data['image']
    targetClasses = data['label']

    inputImages = inputImages.to(device)

    predLogits = model(inputImages).logits
    predLabel = predLogits.argmax(-1).to('cpu').tolist()

    del inputImages, predLogits
    torch.cuda.empty_cache()
    gc.collect()

    allPredictions.extend(predLabel)
    allTargets.extend(targetClasses.tolist())

fireAccuracy = accuracy_score(allTargets, allPredictions)

print('Fire Risk Accuracy =', fireAccuracy)

allFireRisk = pd.DataFrame({'Predicted Class': allPredictions, 'Target Class': allTargets})
allFireRisk.to_csv(RESULTS_PATH + 'Benchmark_Fire_Risk.csv')

confMatrix = confusion_matrix(y_true = allTargets, y_pred = allPredictions)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')

plt.savefig(RESULTS_PATH + 'Benchmark_Fire_Risk_Confusion_Matrix.png')

print('Fire Risk Classification Finished!')

# Game Classification

allPredictions = []
allTargets = []

for _, data in enumerate(gameTest, 0):

    inputImages = data['image']
    targetClasses = data['label']

    inputImages = inputImages.to(device)

    predLogits = model(inputImages).logits
    predLabel = predLogits.argmax(-1).to('cpu').tolist()

    del inputImages, predLogits
    torch.cuda.empty_cache()
    gc.collect()

    allPredictions.extend(predLabel)
    allTargets.extend(targetClasses.tolist())

gameAccuracy = accuracy_score(allTargets, allPredictions)

print('Game Accuracy =', gameAccuracy)

allGame = pd.DataFrame({'Predicted Class': allPredictions, 'Target Class': allTargets})
allGame.to_csv(RESULTS_PATH + 'Benchmark_Game.csv')

confMatrix = confusion_matrix(y_true = allTargets, y_pred = allPredictions)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')

plt.savefig(RESULTS_PATH + 'Benchmark_Game_Confusion_Matrix.png')

print('Game Classification Finished!')

accuracyDF = pd.DataFrame({'Dataset': ['Fire Risk', 'Game'], 
                           'Accuracy': [fireAccuracy, gameAccuracy]})
accuracyDF.to_csv(RESULTS_PATH + 'Benchmark_Accuracies.csv')
