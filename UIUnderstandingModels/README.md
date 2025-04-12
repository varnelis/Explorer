
# Training UI Understanding Computer Vision Models
Here we present the architecture, training, and full annotated datasets (KhanAcademy, Spotify, Wikipedia, WolframAlpha) for agentic UI Understanding.

Our datasets on the KhanAcademy, Spotify, Wikipedia, and WolframAlpha UIs are automatically scraped and annotated by our Selenium-based website scraper and Kotlin-based Android scraper (for Spotify). Our Selenium-based website scraper is outlined in ```ExplorerModelTraining\Explorer```, while our Android scraper is published separately in our *AndroidGUICollection* repository.

This directory is separated into complete sub-directories for dataloading and training Interactable Detection & Screen Similarity models for each of our datasets. We optionally finetune the Interactable Detectors on the final weights trained by Wu et al. in the [WebUI study](https://uimodeling.github.io/).

## Model Training
We trained all models on the [JADE Slurm GPU cluster](https://docs.jade.ac.uk/en/latest/index.html).

We trained Interactable Detector models with multi-GPU training via PyTorch Lightning on 8 GPUs (*distributed data-parallel*) at 32-bit precision. Aggregate learning rate is 0.64 and total batch size is 256; hence, for PT Lightning multi-GPU training, the effective per-GPU LR=0.08 and B=32.

We trained Screen Similarity models on a single GPU at 16-bit precision. Aggregate LR=0.0128 and B=64.

## Training structure
1. A dataloading script loads batches of data for interactable detection or screen similarity.
2. A main training script initializes our FCOS interactable-detection or Siamese Network screen-similarity model architecture, respectively, and trains it using PyTorch Lightning on our datasets.
3. A shell-script is used to submit and run the main training script via Slurm in the JADE cluster.

## Model Weights
We provide the final weights for our Interactable Detection and Screen Similarity models for reproducibility.

The weights are stored in Google Drive as PyTorch ```.ckpt``` files. You can use the ```gdown``` Python package to download them, as implemented in our ```_gdown_url``` method of ```ExplorerModelTraining\Explorer\overlay\shortlister.py```.

### Interactable Detection

Our final trained weights on our auto-collected and labelled datasets are as follows:
- ```khan_interactable_best.ckpt``` ([Google Drive](https://drive.google.com/file/d/1QqV_WjAak43r4NHfpbwioVC5ZVhzMoul/view?usp=sharing)): Final trained FCOS weights for KhanAcademy (```UIUnderstandingModels\models_Khan\screenrecognition```). Used in:
    - KhanAcademy Screen Similarity (```UIUnderstandingModels\models_Khan\screensim```) to give the FCOS FPN features used by the Siamese network for Screen Similarity prediction
    - Can be used as initial weights to finetune FCOS Interactable Detectors for Spotify, Wikipedia, and Wolfram Alpha (```UIUnderstandingModels\models_*\screenrecognition\ui_train_*_finetune.py```)
- ```spotify_interactable_best.ckpt``` ([Google Drive](https://drive.google.com/file/d/1leDymG_L3_0eHTSLLeIOh0llAFRemKeI/view?usp=sharing)): Final trained FCOS weights for Spotify (```UIUnderstandingModels\models_Spotify\screenrecognition```)
- ```screenrecognition_best_res1600-wikipedia.ckpt``` ([Google Drive](https://drive.google.com/file/d/1ydvW29dG04xNhh5xC8BUHAAMcNsz2fZE/view?usp=sharing)): Final trained FCOS weights for Wikipedia (```UIUnderstandingModels\models_Wikipedia\screenrecognition```)
- ```screenrecognition_best_res1600-wolfram.ckpt``` ([Google Drive](https://drive.google.com/file/d/1UlktMQwDfnkQc3OmssnZiuvgiXprEQJO/view?usp=sharing)): Final trained FCOS weights for Wolfram-Alpha (```UIUnderstandingModels\models_Wolfram\screenrecognition```)

Our reproduced trained weights for the [WebUI dataset](https://uimodeling.github.io/) and the [VINS dataset](https://github.com/sbunian/VINS), used as initial weights for finetuning our detectors (KhanAcademy, Spotify, Wikipedia, Wolfram) are as follows:
- ```screenrecognition-web7kbal.ckpt``` ([Google Drive](https://drive.google.com/file/d/1QQVmG6u4jgmptT-iMJdS_ESdEWwuC9U2/view?usp=sharing)): Trained FCOS weights on the Web7kbal dataset from Wu et al. (7k subset of WebUI 350k dataset, with label-cleaning & pre-processing)
- ```screenrecognition-web350k.ckpt``` ([Google Drive](https://drive.google.com/file/d/1WwgONDUkrQSc8NwokL1ePJ_OA3NQh17t/view?usp=sharing)): Trained FCOS weights on full 350k WebUI dataset from Wu et al.; trained from scratch.
- ```screenrecognition-web350k-vins.ckpt``` ([Google Drive](https://drive.google.com/file/d/16a-_TKxAaVYTuWeAdTJVWNW5LXLBeNuY/view?usp=sharing)): Trained FCOS weights on full 350k WebUI dataset from Wu et al.; finetuned from original weights tuned to VINS dataset by Wu et al.

### Screen Similarity
- ```screensim-noocr-spotify-best-EVAL.ckpt``` ([Google Drive](https://drive.google.com/file/d/1cPbA4o0vJwhrM7LVnPs5GPIPmczx5TOG/view?usp=sharing)): Final trained Siamese network weights for Spotify (```UIUnderstandingModels\models_Spotify\screensim```)