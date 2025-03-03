if __name__ == "__main__":

    CHECK_INTERVAL_STEPS = 100

    ARTIFACT_DIR = "./checkpoints_screenrecognition_wolfram"

    import os
    import sys

    # get params for system
    try:
        gpu_num = int(sys.argv[1])
        prec = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        lr = float(sys.argv[4])
        subsize = int(sys.argv[5])
        min_size = int(sys.argv[6])
        max_size = int(sys.argv[7])
    except:
        gpu_num = 8
        prec = 32
        batch_size = 32
        lr = 0.08
        subsize = -1
        min_size = 320
        max_size = 640
    print('Sys Config\n===========================================\n')
    print(f'Batch {batch_size} (effective {batch_size*gpu_num}), LR {lr} (effective {lr/gpu_num}), Data Subset Size {subsize}, DDP optimisations ON\n')
    print(f'Resolution Min Size {min_size}, Max Size {max_size}\n')
    print('=============================================\n\n')

    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    from ui_dataset_wolfram import *
    from ui_models_distributed import *
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import *
    from pytorch_lightning.strategies import DDPStrategy
    import torch
    import datetime
    from time import time
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(ARTIFACT_DIR, version=f'version-res{min_size}')
    
    FINETUNE_CLASSES = 2
    class_weights = [0,1] # no weight on BACKGROUND class, all on clickable class

    #web7kbal_ckpt = os.path.join(ARTIFACT_DIR, 'screenrecognition-web7kbal.ckpt') # load from web7kbal
    #vins_ckpt = os.path.join(ARTIFACT_DIR, 'screenrecognition-vins.ckpt') # load from vins
    #model = UIElementDetector.load_from_checkpoint(web7kbal_ckpt, val_weights=class_weights, test_weights=class_weights, lr=lr)
    #model.hparams.num_classes = FINETUNE_CLASSES
    #print(f'\nMODEL LOADED from {web7kbal_ckpt}\n')

    #model = UIElementDetector(num_classes=2, min_size=min_size, max_size=max_size, lr=lr, val_weights=class_weights, test_weights=class_weights)

    #mod = model.model.head.classification_head
    #model.model.head.classification_head.cls_logits = torch.nn.Conv2d(mod.cls_logits.in_channels, mod.num_anchors * FINETUNE_CLASSES, kernel_size=3, stride=1, padding=1)
    #model.model.head.classification_head.num_classes = FINETUNE_CLASSES
    #model.hparams.num_classes = FINETUNE_CLASSES
    #model.hparams.lr = lr
    
    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")

    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename=f"screenrecognition_best_res{min_size}-wolfram",monitor='val_mAP', mode="max", save_top_k=1)
    
    earlystopping_callback = EarlyStopping(monitor="val_mAP", mode="max", patience=10)

    train_mAP = []
    val_mAP = []
    test_mAP = []
    train_times = []
    test_fps = [] # inference time

    for seed in range(1):

        model = UIElementDetector(num_classes=2, min_size=min_size, max_size=max_size, lr=lr, val_weights=class_weights, test_weights=class_weights)
        #wolfram_ckpt = os.path.join(ARTIFACT_DIR, 'web7kbal.ckpt')
        #model = UIElementDetector.load_from_checkpoint(wolfram_ckpt, val_weights=class_weights, test_weights=class_weights, lr=lr)

        evaluator = Trainer(
            gpus=1,
            precision=prec,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
            min_epochs=1,
            logger=logger,
        )

        trainer_optim = Trainer(
            gpus=gpu_num,
            strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=False), # optimised DDP
            precision=prec, # 16bit precision leads to overflow in MAP calc when running Lightning (srun) (not w/o srun)
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
            min_epochs=150,
            max_epochs=200,
            #max_time='00:05:00:00',
            logger=logger,
            log_every_n_steps=5,
        )

        #if 0 and os.path.exists(os.path.join(ARTIFACT_DIR, 'last-v8.ckpt')): # path DISABLED for now -- not loading from past version
        #    print('loaded from last-v8')
        #    model = UIElementDetector.load_from_checkpoint(
        #        os.path.join(ARTIFACT_DIR, "last-v8.ckpt"), val_weights=class_weights, test_weights=class_weights, lr=lr,
        #    )
        #    model.hparams.lr = lr
        #    model.hparams.num_classes = FINETUNE_CLASSES
        #    mod.num_classes = FINETUNE_CLASSES

        data = WolframUIDataModule(batch_size=batch_size, subset_size=subsize, rand_seed=seed)
        dataloader_train = data.train_dataloader()
        dataloader_test = data.test_dataloader()
        print(f'\nTRAIN DATALOADER SIZE  :  {len(dataloader_train.dataset)}\n')
    
        print('Optimised Trainer..........................................\n')
        train_t0 = time()
        trainer_optim.fit(model, data)
        train_time = time() - train_t0
        print('TRAIN TIME = ', train_time, ' sec\n')
        print('#######################################\n')
    
        print('\nVALIDATING fitted model..................................\n')
        val_dicts = evaluator.validate(model=model, dataloaders=data.val_dataloader())

        print('\nTESTING fitted model.....................................\n')
        test_dicts = evaluator.test(model=model, dataloaders=data.test_dataloader())

        print('\nTRAINING eval mAP on fitted model........................\n')
        train_dicts = evaluator.validate(model=model, dataloaders=data.train_dataloader())
        
        ### FPS CALCULATION ###
        data.batch_size = 1
        test_dicts_fps = evaluator.test(model=model, dataloaders=data.test_dataloader())

        print('DEBUG -- train_dicts    : ', train_dicts)
        print('DEBUG -- val_dicts      : ', val_dicts)
        print('DEBUG -- test_dicts     : ', test_dicts)
        print('DEBUG -- test_dicts_fps : ', test_dicts_fps)

        ## Append results to metric lists
        train_mAP.append(train_dicts[0]['val_mAP'])
        val_mAP.append(val_dicts[0]['val_mAP'])
        test_mAP.append(test_dicts[0]['test_mAP'])
        train_times.append(train_time)
        test_fps.append(len(dataloader_test.dataset) / test_dicts_fps[0]['test_time'])

        print("TRAIN mAP Runs: ", train_mAP)
        print("VAL mAP Runs: ", val_mAP)
        print("TEST mAP Runs: ", test_mAP)
        print("TEST FPS Runs: ", test_fps)


        print("\nMean +/- SD for mAP:")
        print(f"TRAIN = {np.array(train_mAP).mean()} +/- {np.array(train_mAP).std()}")
        print(f"VAL   = {np.array(val_mAP).mean()} +/- {np.array(val_mAP).std()}")
        print(f"TEST  = {np.array(test_mAP).mean()} +/- {np.array(test_mAP).std()}")
        print(f"FPS   = {np.array(test_fps).mean()} +/- {np.array(test_fps).std()}")
        print(f"Tr t  = {np.array(train_times).mean()} +/- {np.array(train_times).std()}")

