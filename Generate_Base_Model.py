import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import BeitForImageClassification, BeitImageProcessor

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import datasets
import gc

torch.manual_seed(3407)
np.random.seed(3407)

EPOCHS = 10
LR = 0.0002
BS = 16
STEPS = 4
BETAS = (0.9, 0.999)
EPSILON = 1e-8


FIRE_PATH = './fire_dataset'
GAME_PATH = './game_dataset'
RESULTS_PATH = './Results/Finetune/'
MODEL_PATH = './Results/Base_Models/'
FIRE_CHECKPOINT = MODEL_PATH + 'fire_base.pt'
GAME_CHECKPOINT = MODEL_PATH + 'game_base.pt'
BEIT_MODEL = 'microsoft/beit-large-patch16-224'

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

def runOneEpoch(trainFlag, dataloader, model, optimizer, scheduler, criterion, accumulationSteps):

    torch.set_grad_enabled(trainFlag)
    if trainFlag:

        model.train()
        for name, param in model.named_parameters():

            if 'classifier' in name:

                param.requires_grad = True
            
            else:

                param.requires_grad = False

    else:

        model.eval()

    losses = []
    allPredictions = []
    allTargets = []

    for index, data in enumerate(dataloader, 0):

        inputImages = data['image'].to(device)
        targetClasses = data['label']
        
        outputs = model(pixel_values = inputImages)

        del inputImages
        torch.cuda.empty_cache()
        gc.collect()

        predLogits = outputs.logits.to('cpu')
        predLabels = predLogits.argmax(-1).tolist()
        
        loss = criterion(predLogits, targetClasses)
        
        allPredictions.extend(predLabels)
        allTargets.extend(targetClasses.tolist())

        if trainFlag:
            
            loss = loss / accumulationSteps
            loss.backward()

            if ((index + 1) % accumulationSteps == 0) or (index + 1 == len(dataloader)):

                optimizer.step()
                optimizer.zero_grad()
        
        losses.append(loss.item())
    
    if trainFlag:
        
        scheduler.step()

    accuracy = accuracy_score(allTargets, allPredictions)

    return np.mean(losses), accuracy

def train(trainLoader, valLoader, model, optimizer, scheduler, criterion, numEpochs, accumulationSteps, checkpoint):

    bestValLoss = np.inf

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []

    for epoch in range(numEpochs):

        print('Epoch', epoch + 1)
        
        trainLoss, trainAccuracy = runOneEpoch(True, trainLoader, model, optimizer, scheduler, criterion, accumulationSteps)
        valLoss, valAccuracy = runOneEpoch(False, valLoader, model, optimizer, scheduler, criterion, accumulationSteps)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)
        trainAccuracies.append(trainAccuracy)
        valAccuracies.append(valAccuracy)

        if valLoss <= bestValLoss:

            torch.save(model.state_dict(), checkpoint)
            bestValLoss = valLoss

        print(f'Training: Loss = {trainLoss} | Accuracy = {trainAccuracy}')
        print(f'Validation: Loss = {valLoss} | Accuracy = {valAccuracy}')

    model.load_state_dict(torch.load(checkpoint))

    return trainLosses, valLosses, trainAccuracies, valAccuracies

def test(dataloader, model):

    torch.set_grad_enabled(False)
    model.eval()

    allPredictions = []
    allTargets = []

    for _, data in enumerate(dataloader, 0):

        inputImages = data['image'].to(device)
        targetClasses = data['label']

        predLogits = model(inputImages).logits
        predLabel = predLogits.argmax(-1).to('cpu').tolist()

        del inputImages
        torch.cuda.empty_cache()
        gc.collect()
        

        allPredictions.extend(predLabel)
        allTargets.extend(targetClasses.tolist())

    accuracy = accuracy_score(allTargets, allPredictions)
    predLabelDict = {'Predicted Class': allPredictions, 'Target Class': allTargets}
    
    return predLabelDict, accuracy

fire = datasets.DatasetDict.load_from_disk('./fire_dataset')
game = datasets.DatasetDict.load_from_disk('./game_dataset')

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

fireTrain = DataLoader(fire['train'], batch_size = BS, shuffle = True)
fireVal = DataLoader(fire['validation'], batch_size = BS, shuffle = True)
fireTest = DataLoader(fire['test'], batch_size = BS, shuffle = True)

gameTrain = DataLoader(game['train'], batch_size = BS, shuffle = True)
gameVal = DataLoader(game['validation'], batch_size = BS, shuffle = True)
gameTest = DataLoader(game['test'], batch_size = BS, shuffle = True)

print('Data Loaders Created!')

criterion = nn.CrossEntropyLoss()

print('Loss Criterion Created!')

fireModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                       num_labels = 4,
                                                       ignore_mismatched_sizes = True)

for name, param in fireModel.named_parameters():

            if 'classifier' in name:

                param.requires_grad = True
            
            else:

                param.requires_grad = False

fireModel.to(device)

print('Fire Risk ResNet Model Initialized!')

fireOptim = optim.Adam(params = fireModel.parameters(),
                       lr = LR,
                       betas = BETAS,
                       eps = EPSILON)

fireScheduler = LinearLR(fireOptim)

print('Fire Risk Adam Optimizer and Scheduler Created!')

fireTrainLosses, fireValLosses, fireTrainAccuracies, fireValAccuracies = train(fireTrain,
                                                                               fireVal,
                                                                               fireModel,
                                                                               fireOptim,
                                                                               fireScheduler,
                                                                               criterion,
                                                                               EPOCHS,
                                                                               STEPS,
                                                                               FIRE_CHECKPOINT)

print('Fire Risk Model Finetuned and Saved!')

firePred, fireTestAccuracy = test(fireTest, fireModel)

print('Test Accuracy =', fireTestAccuracy)

print('Fire Risk Classification Complete!')

del fireModel
torch.cuda.empty_cache()
gc.collect()

gameModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                       num_labels = 4,
                                                       ignore_mismatched_sizes = True)

for name, param in gameModel.named_parameters():

            if 'classifier' in name:

                param.requires_grad = True
            
            else:

                param.requires_grad = False

gameModel.to(device)

print('Game ResNet Model Initialized!')

gameOptim = optim.Adam(params = gameModel.parameters(),
                       lr = LR,
                       betas = BETAS,
                       eps = EPSILON)

gameScheduler = LinearLR(gameOptim)

print('Game Adam Optimizer and Scheduler Created!')

gameTrainLosses, gameValLosses, gameTrainAccs, gameValAccs = train(gameTrain,
                                                                   gameVal,
                                                                   gameModel,
                                                                   gameOptim,
                                                                   gameScheduler,
                                                                   criterion,
                                                                   EPOCHS,
                                                                   STEPS,
                                                                   GAME_CHECKPOINT)

print('Game Model Finetuned and Saved!')

gamePred, gameTestAccuracy = test(gameTest, gameModel)

print('Test Accuracy =', gameTestAccuracy)

del gameModel
torch.cuda.empty_cache()
gc.collect()