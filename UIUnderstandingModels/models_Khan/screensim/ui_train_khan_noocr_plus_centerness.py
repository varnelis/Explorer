if __name__ == "__main__":

    ARTIFACT_DIR = "checkpoints_screensim_khan"
    CHECK_INTERVAL_STEPS = 10

    import os
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    from khan_dataset_screensim import *
    from ui_models_khan_centerness_FC import *
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import *
    from pytorch_lightning.loggers import TensorBoardLogger

    import sys
    try:
        gpu_num = int(sys.argv[1])
        precision = int(sys.argv[2])
        lr = float(sys.argv[3])
        lambda_ocr = float(sys.argv[4])
        batch_size = int(sys.argv[5])
    except IndexError:
        gpu_num = 1
        precision = 16
        lr = 0.00005
        lambda_ocr = 0
        batch_size = 16
    print("***********************************")
    print(f'CONFIGURATION: lr={lr}, lambda_ocr={lambda_ocr}, batch={batch_size}')
    print("***********************************\n")
    
    logger = TensorBoardLogger(ARTIFACT_DIR)
    
    data = KhanSimilarityDataModule(batch_size=batch_size)

    model = UIScreenEmbedder(lr=lr, lambda_ocr=0)

    print("***********************************")
    print("checkpoints: " + str(os.listdir(ARTIFACT_DIR)))
    print("***********************************\n")
    
    checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, 
                                          every_n_train_steps=CHECK_INTERVAL_STEPS, 
                                          save_last=True)
    checkpoint_callback2 = ModelCheckpoint(dirpath=ARTIFACT_DIR, 
                                           filename= "screensim-noocr", 
                                           save_top_k=1, 
                                           every_n_train_steps=CHECK_INTERVAL_STEPS, 
                                           mode="max", 
                                           monitor="val_f1")
    earlystopping_callback = EarlyStopping(monitor="val_f1", mode="max", patience=50)
    
    evaluator = Trainer(
        gpus=1,
        precision=precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        logger=logger,
        limit_val_batches=50
    )
    trainer_optim = Trainer(
        gpus=gpu_num,
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True), # optimised DDP
        precision=precision, # 16bit precision leads to overflow in MAP calc when running Lightning (srun) (not w/o srun)
        gradient_clip_val=1.0,
        val_check_interval=CHECK_INTERVAL_STEPS,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, checkpoint_callback2, earlystopping_callback],
        logger=logger,
        log_every_n_steps=5,
        limit_val_batches=10
    )
    
    ckpt_path = "last-v18.ckpt"
    if False and os.path.exists(os.path.join(ARTIFACT_DIR, ckpt_path)):
        print("***********************************")
        print(f"Loading from {ckpt_path}")
        print("***********************************\n")
        model = UIScreenEmbedder.load_from_checkpoint(os.path.join(ARTIFACT_DIR, ckpt_path))

    trainer_optim.fit(model, data)
    
    # evaluate perf
    print('\nVALIDATING fitted model..................................\n')
    evaluator.validate(model=model, dataloaders=data.val_dataloader())

    print('\nTESTING fitted model.....................................\n')
    evaluator.test(model=model, dataloaders=data.test_dataloader())

    print('\nTRAINING eval mAP on fitted model........................\n')
    evaluator.validate(model=model, dataloaders=data.train_dataloader())
