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

from ContextGatedMOE import loralib as lora

torch.manual_seed(3407)
np.random.seed(3407)

EPOCHS = 25
LR = 0.0002
BS = 16
STEPS = 4
BETAS = (0.9, 0.999)
EPSILON = 1e-8

LORA_RANK = 16
LORA_ALPHA = 32

FIRE_PATH = './fire_dataset'
GAME_PATH = './game_dataset'
RESULTS_PATH = './Results/Lora_Finetune/'
BASE_MODEL_PATH = './Results/Base_Models/'
MODEL_PATH = './Results/Lora_Finetune/Models/'
FIRE_BASE_CHECKPOINT = BASE_MODEL_PATH + 'fire_base.pt'
GAME_BASE_CHECKPOINT = BASE_MODEL_PATH + 'game_base.pt'
FIRE_CHECKPOINT = MODEL_PATH + 'fire_lora_finetune.pt'
GAME_CHECKPOINT = MODEL_PATH + 'game_lora_finetune.pt'
BEIT_MODEL = 'microsoft/beit-large-patch16-224'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def applyLora(model, numLayers, rank, alpha):

    for layer in range(numLayers):

        model.beit.encoder.layer[layer].attention.attention.query = lora.Linear(model.beit.encoder.layer[layer].attention.attention.query.in_features,
                                                                                model.beit.encoder.layer[layer].attention.attention.query.out_features,
                                                                                r = rank,
                                                                                lora_alpha = alpha,
                                                                                bias = False)
        model.beit.encoder.layer[layer].attention.attention.key = lora.Linear(model.beit.encoder.layer[layer].attention.attention.key.in_features,
                                                                              model.beit.encoder.layer[layer].attention.attention.key.out_features,
                                                                              r = rank,
                                                                              lora_alpha = alpha,
                                                                              bias = False)
        model.beit.encoder.layer[layer].attention.attention.value = lora.Linear(model.beit.encoder.layer[layer].attention.attention.value.in_features,
                                                                                model.beit.encoder.layer[layer].attention.attention.value.out_features,
                                                                                r = rank,
                                                                                lora_alpha = alpha,
                                                                                bias = False)
        
        model.beit.encoder.layer[layer].attention.output.dense = lora.Linear(model.beit.encoder.layer[layer].attention.output.dense.in_features,
                                                                             model.beit.encoder.layer[layer].attention.output.dense.out_features,
                                                                             r = rank,
                                                                             lora_alpha = alpha,
                                                                             bias = False)
        
        model.beit.encoder.layer[layer].intermediate.dense = lora.Linear(model.beit.encoder.layer[layer].intermediate.dense.in_features,
                                                                         model.beit.encoder.layer[layer].intermediate.dense.out_features,
                                                                         r = rank,
                                                                         lora_alpha = alpha,
                                                                         bias = False)
        model.beit.encoder.layer[layer].output.dense = lora.Linear(model.beit.encoder.layer[layer].output.dense.in_features,
                                                                   model.beit.encoder.layer[layer].output.dense.out_features,
                                                                   r = rank,
                                                                   lora_alpha = alpha,
                                                                   bias = False)
    
    model.classifier = lora.Linear(model.classifier.in_features,
                                   model.classifier.out_features,
                                   r = rank,
                                   lora_alpha = alpha,
                                   bias = False)
    
    return model

def runOneEpoch(trainFlag, dataloader, model, optimizer, scheduler, criterion, accumulationSteps):

    torch.set_grad_enabled(trainFlag)
    if trainFlag:

        model.train()
        lora.mark_only_lora_as_trainable(model)
    
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

            torch.save(lora.lora_state_dict(model), checkpoint)
            bestValLoss = valLoss

        print(f'Training: Loss = {trainLoss} | Accuracy = {trainAccuracy}')
        print(f'Validation: Loss = {valLoss} | Accuracy = {valAccuracy}')

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
fireVal = DataLoader(fire['validation'], batch_size = BS, shuffle = False)
fireTest = DataLoader(fire['test'], batch_size = BS, shuffle = False)

gameTrain = DataLoader(game['train'], batch_size = BS, shuffle = True)
gameVal = DataLoader(game['validation'], batch_size = BS, shuffle = False)
gameTest = DataLoader(game['test'], batch_size = BS, shuffle = False)

print('Data Loaders Created!')

criterion = nn.CrossEntropyLoss()

print('Loss Criterion Created!')

fireBaseModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                           num_labels = 4,
                                                           ignore_mismatched_sizes = True)

modelStages = len(fireBaseModel.beit.encoder.layer)
fireModel = applyLora(fireBaseModel,
                      numLayers = modelStages,
                      rank = LORA_RANK,
                      alpha = LORA_ALPHA)

fireModel.to(device)
fireModel.load_state_dict(torch.load(FIRE_BASE_CHECKPOINT), strict = False)

print('Fire Risk BEiT Model Initialized!')

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

del fireModel
torch.cuda.empty_cache()
gc.collect()

print('Fire Risk Model Finetuned and Saved!')

fireBaseTestModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                               num_labels = 4,
                                                               ignore_mismatched_sizes = True)

modelStages = len(fireBaseTestModel.beit.encoder.layer)
fireTestModel = applyLora(fireBaseTestModel,
                          numLayers = modelStages,
                          rank = LORA_RANK,
                          alpha = LORA_ALPHA)

fireTestModel.to(device)

fireTestModel.load_state_dict(torch.load(FIRE_BASE_CHECKPOINT), strict = False)
fireTestModel.load_state_dict(torch.load(FIRE_CHECKPOINT), strict = False)

firePred, fireTestAccuracy = test(fireTest, fireTestModel)

print('Test Accuracy =', fireTestAccuracy)

confMatrix = confusion_matrix(firePred['Target Class'],
                              firePred['Predicted Class'])
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')
plt.savefig(RESULTS_PATH + 'Lora_Fire_Risk_Confusion_Matrix.png')

allFireRisk = pd.DataFrame({'Predicted Class': firePred['Predicted Class'],
                            'Target Class': firePred['Target Class']})
allFireRisk.to_csv(RESULTS_PATH + 'Lora_Fire_Risk.csv')

print('Fire Risk Classification Complete!')

del fireTestModel
torch.cuda.empty_cache()
gc.collect()

gameBaseModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                           num_labels = 4,
                                                           ignore_mismatched_sizes = True)

modelStages = len(gameBaseModel.beit.encoder.layer)
gameModel = applyLora(gameBaseModel,
                      numLayers = modelStages,
                      rank = LORA_RANK,
                      alpha = LORA_ALPHA)

gameModel.to(device)
gameModel.load_state_dict(torch.load(GAME_BASE_CHECKPOINT), strict = False)

print('Game BEiT Model Initialized!')

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

del gameModel
torch.cuda.empty_cache()
gc.collect()

print('Game Model Finetuned and Saved!')

gameBaseTestModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                               num_labels = 4,
                                                               ignore_mismatched_sizes = True)

modelStages = len(gameBaseTestModel.beit.encoder.layer)
gameTestModel = applyLora(gameBaseTestModel,
                          numLayers = modelStages,
                          rank = LORA_RANK,
                          alpha = LORA_ALPHA)

gameTestModel.to(device)

gameTestModel.load_state_dict(torch.load(GAME_BASE_CHECKPOINT), strict = False)
gameTestModel.load_state_dict(torch.load(GAME_CHECKPOINT), strict = False)

gamePred, gameTestAccuracy = test(gameTest, gameTestModel)

print('Test Accuracy =', gameTestAccuracy)

confMatrix = confusion_matrix(gamePred['Target Class'],
                              gamePred['Predicted Class'])
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')
plt.savefig(RESULTS_PATH + 'Lora_Game_Confusion_Matrix.png')

allGame = pd.DataFrame({'Predicted Class': gamePred['Predicted Class'],
                        'Target Class': gamePred['Target Class']})
allGame.to_csv(RESULTS_PATH + 'Lora_Game.csv')

print('Game Classification Complete!')

del gameTestModel
torch.cuda.empty_cache()
gc.collect()

plt.figure()

plt.plot(fireTrainAccuracies, label = 'Training')
plt.plot(fireValAccuracies, label = 'Validation')

plt.title('Finetuned LoRA Fire Risk Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(RESULTS_PATH + 'Lora_Fire_Risk_Accuracies.png')

plt.figure()

plt.plot(fireTrainLosses, label = 'Training')
plt.plot(fireValLosses, label = 'Validation')

plt.title('Finetuned LoRA Fire Risk Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(RESULTS_PATH + 'Lora_Fire_Risk_Losses.png')

plt.figure()

plt.plot(gameTrainAccs, label = 'Training')
plt.plot(gameValAccs, label = 'Validation')

plt.title('Finetuned LoRA Game Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(RESULTS_PATH + 'Lora_Game_Accuracies.png')

plt.figure()

plt.plot(gameTrainLosses, label = 'Training')
plt.plot(gameValLosses, label = 'Validation')

plt.title('Finetuned LoRA Game Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(RESULTS_PATH + 'Lora_Game_Losses.png')