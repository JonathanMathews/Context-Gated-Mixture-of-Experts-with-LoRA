# Context Gated Mixture of Experts with LoRA

This project contains all the code to construct a context gated mixture of experts model (formulated as a 2-layer multi-layer perceptron) using multiple low rank adaptation (LoRA) models. This project focuses on finetuning Bidirectional Encoder representation from Image Transformers (BEiT) models for two different tasks of fire-risk and video game image classification.

### Data

The dataset for fire-risk assessment can be found at Hugging Face [here](https://huggingface.co/datasets/blanchon/FireRisk) and the dataset for video games can be found at Kaggle [here](https://www.kaggle.com/datasets/aditmagotra/gameplay-images). Both datasets can be easily downloaded in python using the ```datasets``` library.

Both datasets have a total of 4000 images with each image assigned to one of four total classes (per dataset).

### Setting up Environment

Before running any experiments, first you must make sure the environment is setup correctly. Follow these steps to prepare your system for running the experiments:

1. Clone this repository to your local system, including the submodule ContextGatedMOE, as this contains the source code for the final context gated mixture of LoRAs model.

2. Install the necessary libraries from the ```requirements.txt``` file by running the following line of code in the  terminal:

   ```pip install -r requirements.txt```

3. Create a ```Results``` folder in the main folder that you cloned this repository into. Create specific subfolders so that the folder structure looks like the following (. represents the main folder that you cloned this repository into).

- .
   - Results
      - Base_Models
      - Benchmark
      -  Finetune
         - Models
      - Lora_Finetune
         - Models
      - CGM_Lora
         - Models

Now you are ready to run some experiments!

### Experiments

Each experiment can be run by a single command in the terminal. Each command should be run only while the present working directory is the main folder you cloned this repository into. Each experiment will generate loss graphs for training/validation (except for the benchmark experiment), CSV files containing predicted and actual classes for a withheld test set, as well as checkpoint PyTorch models during finetuning. Follow the steps below to run the experiments and evaluate the results of all the models:

1. **Prepare Data**

   Run the following line in the terminal:

   ```python Data_Preparation.py```

   This will create two folders that contain Python Dataset objects for both the fire-risk and video game datasets. These folders will contain the data used in all future experiments.
   
3. **Generate Base Models for LoRA and CGM**

   Run the following line in the terminal:

   ```python Generate_Base_Model.py```

   This will create base PyTorch models for the two datasets to use as pre-trained weight loaders for training LoRA and CGM fine-tunes in later experiments.
   
4. **Benchmark Data**

   Run the following line in the terminal:

   ```python Benchmark.py```

   This will create two CSV files in the ```Results/Benchmark_Data``` folder containing predicted and actual classes for fire-risk and video game datsets, respectively. It will also create two confusion matrices in the same location that show the predicted vs. actual results of classification on each dataset.

5. **Full Finetune**

   Run the following line in the terminal:

   ```python Finetune.py```

   This will create two PyTorch models in the ```Results/Finetune/Models``` folder representing the two fully finetuned models, two CSV files in the ```Results/Finetune``` folder containing predicted and actual classes for fire-risk and video game datsets, two confusion matrices that show the predicted vs. actual results of classification on each dataset, and two training/validation loss graphs and two training/validation accuracy graphs in the ```Results/Finetune``` folder.

6. **LoRA Finetune**

   Run the following line in the terminal:

   ```python Lora_Finetune.py```

   This will create two PyTorch models in the ```Results/Lora_Finetune/Models``` folder representing the two LoRA finetuned models, two CSV files in the ```Results/Lora_Finetune``` folder containing predicted and actual classes for fire-risk and video game datsets, two confusion matrices that show the predicted vs. actual results of classification on each dataset, and two training/validation loss graphs and two training/validation accuracy graphs in the ```Results/Lora_Finetune``` folder.

7. **Context Gated Mixture of LoRA Models (CGM LoRA)**

   Run the following line in the terminal:

   ```python CGM_Lora.py```

   This will create one PyTorch model in the ```Results/CGM_Lora/Models``` folder representing the context gated mixture of LoRA model, two CSV files in the ```Results/CGM_Lora``` folder containing predicted and actual classes for fire-risk and video game datsets, two confusion matrices that show the predicted vs. actual results of classification on each dataset, and one training/validation loss graph and one training/validation accuracy graph in the ```Results/CGM_Lora``` folder.
