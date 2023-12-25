
import json, math
import numpy as np

def rand_train_split_Manual():

    ALL_FILE = 'train_split_web7k.json'
    TRAIN_FILE = 'train_split_web7k_Manual.json'
    VAL_FILE = 'val_split_web7k_Manual.json'
    TEST_FILE = 'test_split_web7k_Manual.json'

    with open(ALL_FILE, 'r') as fp:
        all_data = np.array(json.load(fp))

    # 7000 all -> 90% train, 5% val, 5% test
    perm = np.random.permutation(all_data)

    val_break = math.floor(len(all_data)*0.9)
    test_break = math.floor(len(all_data)*0.95)

    train = list(perm[0:val_break])
    val = list(perm[val_break:test_break])
    test = list(perm[test_break:])

    with open(TRAIN_FILE, 'w') as fp:
        json.dump(train, fp)
    with open(VAL_FILE, 'w') as fp:
        json.dump(val, fp)
    with open(TEST_FILE, 'w') as fp:
        json.dump(test, fp)

if __name__ == '__main__':
    rand_train_split_Manual()