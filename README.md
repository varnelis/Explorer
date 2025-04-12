# The Explorer

Cross platform system for GUI automation using Machine Vision. This repository is part of the paper 'name and link' released in arXiv, and therefore it is highly advisable to first read the paper before using this repository.
  
## 🧐 Features

Here're some of the project's best features:

*   Cellular automaton inspired interactable detection algorithm
*   Automated data collection and labelling pipeline
    * Selenium-based collection for computer websites
    * Kotlin-based collection for Android apps (see our [AndroidGUICollection](https://github.com/IasonC/AndroidGUICollection) repository)
*   Crawler and analysis tool for KhanAcademy
*   Screen and keyboard recording, analysing, and saving tools
*   Machine Vision models to identify clickable GUI areas and changes in GUI state
*   Backend with MongoDB (you can replace with your api-key )
*   Tool for analysing OCR'ed data for page similarity matching
*   Voice-powered GUI navigation
*   "Trace" creation tool (as explained in the paper)
*   "Trace" replication tool (as explained in the paper)
*   "Action matching" tool (as explained in the paper)

## 🛠️ Installation Steps:
This repository uses Poetry for python package management. See poetry documentation at https://python-poetry.org/docs/.
Most of the Features of the project you can access through CLI defined in `./main.py`. Example:
`python main.py hello-world`

## 🧠 Machine Vision Models

In this paper we implement Machine Vision models for Interactable Detection (FCOS object detector) and Screen Similarity detection (Siamese net). These enable our end-to-end features of robust and platform-independent (1) voice-powered GUI navigation and (2) "Trace" replication.

See directory ```UIUnderstandingModels``` of branch ```Add/ModelTraining``` for the model implementation (PyTorch Lightning) and training. This directory also includes our complete datasets of GUIs across computer websites (KhanAcademy, Wikipedia, Wolfram-Alpha) and Android apps (Spotify).

## 🛡️ License:

This project is licensed under the MIT

## 💖Like our work?

This work was a collaboration between Arnas Vyšniauskas, Iason Chaimalas for Master thesis project at UCL supervised by Dr Alejandra Beghelli and advised by Prof Gabriel Brostow. You can contact us through our project supervisor Prof Gabriel Brostow at gabriel.brostow@ucl.ac.uk
