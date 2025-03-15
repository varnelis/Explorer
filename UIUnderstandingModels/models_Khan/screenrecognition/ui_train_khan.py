if __name__ == "__main__":

    CHECK_INTERVAL_STEPS = 1000

    ARTIFACT_DIR = "./checkpoints_screenrecognition_khan"

    import os
    import sys

    # get params for system
    try:
        gpu_num = int(sys.argv[1])
        prec = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        lr = float(sys.argv[4])
    except:
        gpu_num = 8
        prec = 32
        batch_size = 32
        lr = 0.01
    print('Sys Config\n===========================================\n')
    print(f'Batch {batch_size}, lr {lr}, DDP optimisations ON\n')
    print('=============================================\n\n')

    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    from ui_dataset_khan import *
    from ui_models_distributed import *
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import *
    from pytorch_lightning.strategies import DDPStrategy
    import torch
    import datetime
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(ARTIFACT_DIR)
    
    data = KhanUIDataModule(batch_size=batch_size)

    FINETUNE_CLASSES = 2
    class_weights = [0,1] # no weight on BACKGROUND class, all on clickable class

    model = UIElementDetector(num_classes=2, min_size=1280, max_size=2560, lr=lr, val_weights=class_weights, test_weights=class_weights)
    
    #web7kbal_ckpt = os.path.join(ARTIFACT_DIR, 'screenrecognition-web7kbal.ckpt') # load from web7kbal
    #vins_ckpt = os.path.join(ARTIFACT_DIR, 'screenrecognition-vins.ckpt') # load from vins
    #model = UIElementDetector.load_from_checkpoint(web7kbal_ckpt, val_weights=class_weights, test_weights=class_weights, lr=lr)
    #model.hparams.num_classes = FINETUNE_CLASSES
    #print(f'\nMODEL LOADED from {web7kbal_ckpt}\n')

    #mod = model.model.head.classification_head
    #model.model.head.classification_head.cls_logits = torch.nn.Conv2d(mod.cls_logits.in_channels, mod.num_anchors * FINETUNE_CLASSES, kernel_size=3, stride=1, padding=1)
    #model.model.head.classification_head.num_classes = FINETUNE_CLASSES
    model.hparams.num_classes = FINETUNE_CLASSES
    model.hparams.lr = lr
    
    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************")

    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenrecognition",monitor='val_mAP', mode="max", save_top_k=1)
    
    earlystopping_callback = EarlyStopping(monitor="val_mAP", mode="max", patience=10)

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
        max_epochs=150,
        logger=logger,
        log_every_n_steps=5,
    )

    if os.path.exists(os.path.join(ARTIFACT_DIR, 'khan_interactable_best.ckpt')):
        print('loaded from last-v8')
        model = UIElementDetector.load_from_checkpoint(
            os.path.join(ARTIFACT_DIR, "khan_interactable_best.ckpt"), val_weights=class_weights, test_weights=class_weights, lr=lr,
        )
        model.hparams.lr = lr
        model.hparams.num_classes = FINETUNE_CLASSES
        mod.num_classes = FINETUNE_CLASSES

    print('Optimised Trainer..........................................\n')
    trainer_optim.fit(model, data)
    
    print('\nVALIDATING fitted model..................................\n')
    evaluator.validate(model=model, dataloaders=data.val_dataloader())

    print('\nTESTING fitted model.....................................\n')
    evaluator.test(model=model, dataloaders=data.test_dataloader())

    print('\nTRAINING eval mAP on fitted model........................\n')
    evaluator.validate(model=model, dataloaders=data.train_dataloader())
