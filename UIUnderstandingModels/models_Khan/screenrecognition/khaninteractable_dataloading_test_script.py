from ui_dataset_khan import *

kd = KhanUIDataModule()

kd_train = kd.train_dataset
print("Train len: ", kd_train.__len__())

kd_val = kd.val_dataset
print("Val len: ", kd_val.__len__())

kd_test = kd.test_dataset
print("Test len: ", kd_test.__len__())

