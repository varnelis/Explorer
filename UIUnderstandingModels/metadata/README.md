## metadata
Folder for the metadata of the ```../downloads``` datasets for training UI Understanding neural models.
### Format
Metadata for Interactable Detection (```./screenrecognition```) and Screen Similarity (```./screensim```) tasks contains the class-map file for prediction labels and the JSON files for the dataset UUIDs split into training-validation-test sets.

In **Interactable Detection**:
- ```class_map_*_manual.json``` gives the mapping of prediction labels ```0``` (not interactable) and ```1``` (interactable/clickable) for predicted bounding boxes of the Interactale Detector in our KhanAcademy, Spotify, Wikipedia & Wolfram-Alpha GUI Datasets.
- ```*_ids_*.json``` gives the train/val/test set UUIDs for our  KhanAcademy, Spotify, Wikipedia & Wolfram-Alpha GUI Datasets. The format is ```{"items": List[{"uuid": UUID}]}```.

In **Screen Similarity**:
- ```domain_map*.json``` gives the grouping of GUI screenshots (described by screenshot UUID) into same-state groups (described by group UUID -- webpage URL for KhanAcademy dataset or ```SAMESTATE_<UUID>``` for Spotify dataset) --> from the same-state grouping we infer whether 2 random-sampled GUI screenshots are of the same GUI state (*Do they belong to the same group UUID or not?*)
- ```*_split_*_screensim.json``` gives the group UUIDs used in the train/val/test data splits --> screensim dataloader will sample 2 screenshots at a time from any of the group UUIDs in the same split and pass them to the Siamese network.