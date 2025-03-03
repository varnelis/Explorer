from khan_dataset_screensim import *
kd = KhanSimilarityDataModule()

ktrain = kd.train_dataset
kval = kd.val_dataset
ktest = kd.test_dataset
domain_map = ktrain.domain_map

for dataset in [ktrain,kval,ktest]:
    domains = dataset.domain_list
    datasize = list(map(lambda x: len(domain_map[x]),domains))
    print(sum(datasize))

# NOTE:
# Total dataset is 310 train, 31 val, 37 test = 378 screenshots as reported in the paper
# If we constrain each same-state group to have at least 2 unique screenshots (see ./khan_dataset_screensim.py Line 55) 
# then the dataset is 280 train, 23 val, 35 test = 338 screenshots.