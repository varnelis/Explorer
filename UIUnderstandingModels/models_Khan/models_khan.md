## Interactable Detection models for KhanAcademy dataset

Train interactable detector as FCOS model.

Optionally fine-tune from the public VINS (https://github.com/sbunian/VINS) or WebUI (https://uimodeling.github.io/) datasets by downloading model weights (see Wu et al., 'WebUI', CHI 2023)


## Screen Similarity models for KhanAcademy dataset

### Siamese-based Screen Similarity model training

The code structure is:
- ```khan_dataset_screensim.py``` is the dataloader loading 2 random-sampled GUI screenshots in the train/val/test set along with the ground-truth label: same-state or different-state
- ```ui_models_khan_centerness_FC.py``` & ```ui_models_khan_centerness_FC_EVALALL.py``` train the Siamese network with Centerness augmentation (see paper). This involves call to ```embedding_exploration/Encoder```.
- ```ui_train_khan_noocr_plus_centerness.py``` defines the training loop using PyTorch Lightning
- ```job_Khan_NoOCR_plus_center.sh``` submits training job to JADE SLURM cluster

### Ablation on Siamese Architecture
We tested various Siamese Network architectures, involving a basic Linear layer plus Instance/Batchnorm layers, ReLU activation, NesNet layer, or NesNeXt layer. This is shown in Figure 14 of our paper.

The relevant files for these ablated architectures are (```*``` = ```LinearBase```, ```NoIN```, ```NoReLU```, ```resnet```):
- Siamese architecture ```ui_models_khan_*.py```
- Model training: ```ui_train_khan_noocr_*.py```
- Training job submitted to JADE SLURM cluster: ```job_Khan_NoOCR_*.sh```