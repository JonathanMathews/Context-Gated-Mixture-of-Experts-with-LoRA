import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import BeitImageProcessor, BeitForImageClassification

from tqdm.notebook import tqdm_notebook

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
RESULTS_PATH = './Results/CGM_Lora/'
BASE_MODEL_PATH = './Results/Base_Models/'
MODEL_PATH = './Results/CGM_Lora/Models/'
LORA_PATH = './Results/Lora_Finetune/Models/'
BASE_CHECKPOINT = BASE_MODEL_PATH + 'fire_base.pt'
FIRE_LORA_CHECKPOINT = LORA_PATH + 'fire_lora_finetune.pt'
GAME_LORA_CHECKPOINT = LORA_PATH + 'game_lora_finetune.pt'
CGM_CHECKPOINT = MODEL_PATH + 'fire_game_cgm_lora.pt'
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

def applyLoraCGM(model, numLayers, alpha, fireLoraWeights, gameLoraWeights):

    for i in range(numLayers):

        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.query.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.query.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.query.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.query.lora_B']]
        model.beit.encoder.layer[i].attention.attention.query = lora.CGMLinear(loraAWeights,
                                                                               loraBWeights,
                                                                               lora_alpha = alpha,
                                                                               bias = False)
        
        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.key.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.key.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.key.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.key.lora_B']]
        model.beit.encoder.layer[i].attention.attention.key = lora.CGMLinear(loraAWeights,
                                                                             loraBWeights,
                                                                             lora_alpha = alpha,
                                                                             bias = False)

        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.value.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.value.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.attention.value.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.attention.value.lora_B']]
        model.beit.encoder.layer[i].attention.attention.value = lora.CGMLinear(loraAWeights,
                                                                               loraBWeights,
                                                                               lora_alpha = alpha,
                                                                               bias = False)
        
        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.output.dense.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.output.dense.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.attention.output.dense.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.attention.output.dense.lora_B']]
        model.beit.encoder.layer[i].attention.output.dense = lora.CGMLinear(loraAWeights,
                                                                            loraBWeights,
                                                                            lora_alpha = alpha,
                                                                            bias = False)
        
        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.intermediate.dense.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.intermediate.dense.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.intermediate.dense.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.intermediate.dense.lora_B']]
        model.beit.encoder.layer[i].intermediate.dense = lora.CGMLinear(loraAWeights,
                                                                        loraBWeights,
                                                                        lora_alpha = alpha,
                                                                        bias = False)
        
        loraAWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.output.dense.lora_A'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.output.dense.lora_A']]
        loraBWeights = [fireLoraWeights[f'beit.encoder.layer.{i}.output.dense.lora_B'],
                        gameLoraWeights[f'beit.encoder.layer.{i}.output.dense.lora_B']]
        model.beit.encoder.layer[i].output.dense = lora.CGMLinear(loraAWeights,
                                                                  loraBWeights,
                                                                  lora_alpha = alpha,
                                                                  bias = False)
        
    loraAWeights = [fireLoraWeights[f'classifier.lora_A'],
                    gameLoraWeights[f'classifier.lora_A']]
    loraBWeights = [fireLoraWeights[f'classifier.lora_B'],
                    gameLoraWeights[f'classifier.lora_B']]
    model.classifier = lora.CGMLinear(loraAWeights, loraBWeights, lora_alpha = alpha, bias = False)

    return model

def runOneEpoch(trainFlag, fireDL, gameDL, model, optimizer, scheduler, criterion):

    torch.set_grad_enabled(trainFlag)
    if trainFlag:

        model.train()
        for name, param in model.named_parameters():

            if 'context_gated_mixing' in name:

                param.requires_grad = True
            
            else:

                param.requires_grad = False
        
    else:

        model.eval()

    losses = []
    allPredictions = []
    allTargets = []

    for multiDataBatch in tqdm_notebook(zip(fireDL, gameDL), desc = 'Batches', total = len(gameDL)):

        for batch in multiDataBatch:

            inputImages = batch['image'].to(device)
            targetClasses = batch['label']
            
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
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())
        
    if trainFlag:
        
        scheduler.step()

    accuracy = accuracy_score(allTargets, allPredictions)

    return np.mean(losses), accuracy

def train(fireTrain, fireVal, gameTrain, gameVal, model, optimizer, scheduler, criterion, numEpochs, checkpoint):

    bestValLoss = np.inf

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []

    for epoch in range(numEpochs):

        print('Epoch', epoch + 1)

        trainLoss, trainAccuracy = runOneEpoch(True, fireTrain, gameTrain, model, optimizer, scheduler, criterion)
        valLoss, valAccuracy = runOneEpoch(False, fireVal, gameVal, model, optimizer, scheduler, criterion)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)
        trainAccuracies.append(trainAccuracy)
        valAccuracies.append(valAccuracy)

        if valLoss <= bestValLoss:

            torch.save(lora.cgm_state_dict(model), checkpoint)
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

baseModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                       num_labels = 4,
                                                       ignore_mismatched_sizes = True)

print('Base Model Created!')

fireLoraWeights = torch.load(FIRE_LORA_CHECKPOINT)
gameLoraWeights = torch.load(GAME_LORA_CHECKPOINT)

print('Fire and Game LoRA Weights Loaded')

modelStages = len(baseModel.beit.encoder.layer)
model = applyLoraCGM(baseModel, modelStages, LORA_ALPHA, fireLoraWeights, gameLoraWeights)

model.to(device)
model.load_state_dict(torch.load(BASE_CHECKPOINT), strict = False)

for name, param in model.named_parameters():

    if 'context_gated_mixing' in name:

        param.requires_grad = True

    else:

        param.requires_grad = False

print('Context-Gated Mixing Layers Initialized!')

optimizer = optim.Adam(params = model.parameters(),
                       lr = LR,
                       betas = BETAS,
                       eps = EPSILON)

scheduler = LinearLR(optimizer)

print('Adam Optimizer and Scheduler Created!')

trainLosses, valLosses, trainAccuracies, valAccuracies = train(fireTrain,
                                                               fireVal,
                                                               gameTrain,
                                                               gameVal,
                                                               model,
                                                               optimizer,
                                                               scheduler,
                                                               criterion,
                                                               EPOCHS,
                                                               CGM_CHECKPOINT)

del model
torch.cuda.empty_cache()
gc.collect()

print('CGM Model Trained and Saved!')

baseTestModel = BeitForImageClassification.from_pretrained(BEIT_MODEL,
                                                           num_labels = 4,
                                                           ignore_mismatched_sizes = True)

fireLoraWeights = torch.load(FIRE_LORA_CHECKPOINT)
gameLoraWeights = torch.load(GAME_LORA_CHECKPOINT)

modelStages = len(baseTestModel.beit.encoder.layer)
testModel = applyLoraCGM(baseTestModel, modelStages, LORA_ALPHA, fireLoraWeights, gameLoraWeights)

testModel.to(device)

testModel.load_state_dict(torch.load(BASE_CHECKPOINT), strict = False)
testModel.load_state_dict(torch.load(CGM_CHECKPOINT), strict = False)

firePred, fireTestAccuracy = test(fireTest, testModel)

print('Test Accuracy =', fireTestAccuracy)

confMatrix = confusion_matrix(firePred['Target Class'],
                              firePred['Predicted Class'])
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')
plt.savefig(RESULTS_PATH + 'CGM_Lora_Fire_Risk_Confusion_Matrix.png')

allFireRisk = pd.DataFrame({'Predicted Class': firePred['Predicted Class'],
                            'Target Class': firePred['Target Class']})
allFireRisk.to_csv(RESULTS_PATH + 'CGM_Lora_Fire_Risk.csv')

print('Fire Risk Classification Complete!')

gamePred, gameTestAccuracy = test(gameTest, testModel)

print('Test Accuracy =', gameTestAccuracy)

confMatrix = confusion_matrix(gamePred['Target Class'],
                              gamePred['Predicted Class'])
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot(cmap = 'Purples')
plt.savefig(RESULTS_PATH + 'CGM_Lora_Game_Confusion_Matrix.png')

allGame = pd.DataFrame({'Predicted Class': gamePred['Predicted Class'],
                        'Target Class': gamePred['Target Class']})
allGame.to_csv(RESULTS_PATH + 'CGM_Lora_Game.csv')

print('Game Classification Complete!')

plt.figure()

plt.plot(trainAccuracies, label = 'Training')
plt.plot(valAccuracies, label = 'Validation')

plt.title('Context-Gated Mixing - Fire & Game Classification Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(RESULTS_PATH + 'CGM_Lora_Accuracies.png')

plt.figure()

plt.plot(trainLosses, label = 'Training')
plt.plot(valLosses, label = 'Validation')

plt.title('Context-Gated Mixing - Fire & Game Classification Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(RESULTS_PATH + 'CGM_Lora_Losses.png')