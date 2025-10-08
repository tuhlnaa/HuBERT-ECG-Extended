import os
import argparse
import copy
import random
import wandb
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from transformers import HubertConfig
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from math import ceil

from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from torchmetrics.classification import MulticlassAUROC
from torcheval.metrics import MultilabelAUPRC as AUPRC

# Import custom modules
from config import RichPrinter, create_parser, init_seeds, validate_args
from dataset import ECGDataset
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig
from hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
SUPERVISED_MODEL_CKPT_PATH = "./models/checkpoints/supervised/"
DROPOUT_DYNAMIC_REG_FACTOR = 0.05
    

def dynamic_regularizer(optimizer, model, penalty):
    if penalty:
        # penalizing model with regularization but not too much
        optimizer.param_groups[0]['weight_decay'] *= 5
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p += 0.05
    else:
        # unburdening model from regularization
        # minimum attainable weight decay is 0.01, dropout is 0.1
        optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'] / 5)
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p = max(0.1, module.p - DROPOUT_DYNAMIC_REG_FACTOR)

def finetune(args):
    
    device = torch.device('cuda')
    
    ### NOTE: comment for sweeps, uncomment for normal run ###
    #wandb.init(entity="my-entity", project="my-project", group="supervised")
    wandb.init(project="my-project", group="supervised")

    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name
    init_seeds()
    # torch.manual_seed(42)
    # np.random.seed(42)
    # torch.cuda.manual_seed(42)
    # random.seed(42)
    
    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path="./output/PTB", # "/data/ECG_AF/train_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop
    )

    train_pos_weights = train_set.weights.to(device) if args.use_loss_weights else None

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path="./output/PTB", # "/data/ECG_AF/val_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop
        )
    
    val_pos_weights = val_set.weights.to(device) if args.use_loss_weights else None
    
    train_dl = DataLoader(
        train_set,
        collate_fn=train_set.collate,
        #num_workers=6,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
        )

    val_dl = DataLoader(
        val_set,
        collate_fn=val_set.collate,
        #num_workers=6,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
        )
    
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    
    task2criteria = {
        'multi_class': (torch.nn.CrossEntropyLoss(weight=train_pos_weights), torch.nn.CrossEntropyLoss(weight=val_pos_weights)),
        'multi_label': (torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weights), torch.nn.BCEWithLogitsLoss(pos_weight=val_pos_weights)),
        'regression' : (torch.nn.MSELoss(), torch.nn.MSELoss())
    }    
    
    criterion = task2criteria[args.task]
    criterion_train = criterion[0].to(device)
    criterion_val = criterion[1].to(device)
    
    args.training_steps = args.training_steps if args.training_steps is not None else ((args.epochs - 1) * (len(train_dl) // accumulation_steps))
    
    args.val_interval = len(train_dl) if args.val_interval is None else args.val_interval
    
    logger.info(f"{args.training_steps} training steps to perform")
    logger.info(f"{args.val_interval} steps to wait before validation")
    
    if args.resume_finetuning:
        
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to resume finetuning")
        
        checkpoint = torch.load(args.load_path, map_location = 'cpu', weights_only=False)

        config = checkpoint['model_config']

        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
        config.conv_pos_batch_norm = False
        
        pretrained_hubert = HuBERT(config)
        
        assert checkpoint['finetuning_vocab_size'] == args.vocab_size, "Vocab size mismatch"
        assert checkpoint['use_label_embedding'] == args.use_label_embedding, "Label embedding mismatch"
        assert checkpoint['linear'] == True if args.classifier_hidden_size is None and not args.use_label_embedding else False, "Classifier mismatch"
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.finetuning_vocab_size, classifier_hidden_size=args.classifier_hidden_size, use_label_embedding=args.use_label_embedding)
        hubert.to(device)
        
        hubert.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict false prevents errors when trying to match mask token key
        
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_count = checkpoint['patience_count']
        best_val_target_score = checkpoint[f'target_val_{args.target_metric}']
        
        if args.freezing_steps is not None and global_step < args.freezing_steps:
            hubert.set_transformer_blocks_trainable(n_blocks=0)
            hubert.set_feature_extractor_trainable(False)
        else:
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
            
        # if layuer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
            num_training_steps=args.training_steps,
        )

        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        logger.info("Checkpoint loaded. Model ready to resume finetuning.")
        
    elif args.random_init:
        logger.info("Creating a model for fully supervised training")
        
        ### size model hyperparams ###
        if args.largeness == "base":
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
            intermediate_size = 3072    
            classifier_proj_size = 256
        elif args.largeness == "large":
            hidden_size = 960
            num_hidden_layers = 16
            num_attention_heads = 12
            intermediate_size = 3840
            classifier_proj_size = 512
        elif args.largeness == 'small': # small
            hidden_size = 512
            num_hidden_layers = 8
            num_attention_heads = 8
            intermediate_size = 2048
            classifier_proj_size = 256
        else:
            raise ValueError(f"Model largeness {args.largeness} not supported")
        
        if args.downsampling_factor is None:
            conv_kernel = (10, 3, 3, 3, 3, 2, 2)
            conv_stride = (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512, 512, 512)   
        elif args.downsampling_factor == 5: 
            conv_kernel = (10, 3, 3, 2, 2)
            conv_stride = (4, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512)
        elif args.downsampling_factor == 10:
            conv_kernel = (10, 3, 3, 2)
            conv_stride = (4, 2, 2, 2)
            conv_dim = (512, 512, 512, 512)
        else:
            raise ValueError(f"Downsampling factor {args.downsampling_factor} not supported")
                
        config = HuBERTECGConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            mask_time_prob = 0.0, 
            classifier_proj_size = classifier_proj_size,
            layerdrop = args.finetuning_layerdrop,
            conv_kernel = conv_kernel,
            conv_stride = conv_stride,
            conv_dim = conv_dim,
            mask_time_length = 1,
            hidden_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            feat_proj_dropout=max(0, 0 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),    
        ) # + other default params
        
        hubert = HuBERT(config)
        
        hubert = HuBERTClassification(hubert,
                                      num_labels=args.vocab_size,
                                      classifier_hidden_size=args.classifier_hidden_size,
                                      use_label_embedding=args.use_label_embedding
                                      )
        hubert.to(device)  
        
        
        global_step = 0
        best_val_loss = float("inf")
        best_val_target_score = 0.0
        patience_count = 0  
              
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(0.08*args.training_steps),
            num_training_steps=args.training_steps
        )
        
        logger.info("Model created. Ready for fully supervised training.")
        
    else:
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to start finetuning")
                
        checkpoint = torch.load(args.load_path, map_location = 'cpu', weights_only=False)
        config = checkpoint['model_config']
        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
        config.layerdrop = args.finetuning_layerdrop
        config.conv_pos_batch_norm = False

        pretrained_hubert = HuBERT(config)

        # restore original p-dropout or set multipliers
        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.vocab_size, classifier_hidden_size=args.classifier_hidden_size,  use_label_embedding=args.use_label_embedding)
        hubert.to(device)         

        # Transfer learning: Loading a pretrained model but with a different final layer
        hubert.hubert_ecg.load_state_dict(checkpoint['model_state_dict'], strict=False) # load backbone weights

        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_val_target_score = 0.0
        
        
        # ASSUMPTION: the classification head is always trainable
        # transformer_blocks_to_unfreeze is 0 by default, meaning that if no input comes from the user, the transformer encoder remains frozen
        # if the user wants to unfreeze some blocks, he must provide the number of blocks to unfreeze.
        
        # Freezing the convolutional feature extractor is a precise choice of the user. By default it is frozen, but it can unfrozen by using the flag --unfreeze_conv_embedder
        
        # If the freezing_steps argument is provided, then the model is completely frozen to allow to train only the head for that number of steps.
        # After that, n blocks of the transformer's encoder are unfronzen and the conv feature extractor is unfrozen as well in the training loop if the flag --unfreeze_conv_embedder is used.
        # Freezing_steps is none by default
        
        # Nte: if freezing_steps equals training_steps, then the model is completely frozen for the whole training
        
        if args.freezing_steps is not None:
            hubert.set_transformer_blocks_trainable(n_blocks=0) # freeze all transformer blocks
            hubert.set_feature_extractor_trainable(False) # freeze conv feature extractor
        else:        
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze) # makes trainable only the last n_blocks of the transformer encoder
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder) # frozen by default
        
        # if layer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        # first 8 layer with lower lr and last 4 with higher lr based on Primer Bertology
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            logger.info("Setting layer-wise learning rate")
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )
        
        logger.info("Checkpoint loaded. Model ready for finetuning.")
    
    #scaler = amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda') 

    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs
    
    start_epoch = global_step // len(train_dl)
    
    task2metric = {
        'multi_label': {
                        "f1-score" : F1_score(num_labels=args.vocab_size, average=None),
                        "recall" : Recall(num_labels=args.vocab_size, average=None),
                        "specificity" : Specificity(num_labels=args.vocab_size, average=None),
                        "precision" : Precision(num_labels=args.vocab_size, average=None),
                        "auroc" : MultilabelAUROC(num_labels=args.vocab_size, average=None),
                        "auprc" : AUPRC(num_labels=args.vocab_size, average=None),
        },
        'multi_class': {
                        'accuracy' : Accuracy(num_classes=args.vocab_size),
                        'auroc' : MulticlassAUROC(num_classes=args.vocab_size)
            },
        'regression' : {}
    }
    
    metrics = task2metric[args.task]
    
    assert args.target_metric in metrics.keys(), f"Target metric {args.target_metric} not available for task {args.task}"
    
    for name, metric in metrics.items():
        metric.to(device)
    
    for epoch in range(start_epoch, epochs):
    
        hubert.train()
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_losses = []
        
        for ecg, attention_mask, labels in tqdm(train_dl, total=len(train_dl)):
            
            global_step += 1
            
            ecg = ecg.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.squeeze().to(device)
            
            #with amp.autocast():
            with torch.amp.autocast('cuda'):
                logits, _ = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=False)
                loss = criterion_train(logits, labels)
                
                loss /= accumulation_steps # normalize loss
                
            scaler.scale(loss).backward() # accumulate normalized loss
            train_losses.append(loss.item())                
                
            if global_step % accumulation_steps == 0: 
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                
            if args.freezing_steps is not None and global_step >= args.freezing_steps:
                
                # unfreeze what you wanted to unfreeze, as specificied by user input
                hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
                hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
                
                # recreate optimizer to update params when unfreezing them
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, hubert.parameters()),
                    lr=lr,
                    betas=betas,
                    eps=EPS,
                    weight_decay=weight_decay,
                )
                # recreate lr_scheduler to update params when unfreezing them
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
                    num_training_steps=args.training_steps,
                )
            
            ### validation ###        
            if global_step % args.val_interval == 0:
                
                hubert.eval()
                
                # reset metrics and losses                
                val_losses = []                
                for name, metric in metrics.items():
                    metric.reset()
                
                logger.info("Start validation at step {}".format(global_step))
                
                ### validation loop ###
                for ecg, _, labels in tqdm(val_dl, total=len(val_dl)):
                    
                    ecg = ecg.to(device)
                    labels = labels.squeeze().to(device)
                    
                    with torch.no_grad():
                        logits, _ = hubert(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False)
                        loss = criterion_val(logits, labels)
                    
                    val_losses.append(loss.item())
                    
                    labels = labels.long() # for metrics
                    
                    # compute metrics on single batch
                    for name, metric in metrics.items():
                        metric.update(logits, labels)
                    
                ### end of validation loop ###
                
                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep train loss aligned with val loss
                
                                
                # log non averaged metrics [1, vocab_size]
                logger.info("Validation loss = {}".format(val_loss))
                    
                # compute metrics on whole validation set and log them
                # such metrics are vectors num_labels long containing the metric for each label
                for name, metric in metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    logger.info(f"Validation {name} = {score}, macro: {macro}")
                    wandb.log({f"Validation_{name}": macro})
                    if name == args.target_metric:
                        target_score = macro
                
                # log losses
                wandb.log({
                    "Training_loss": train_loss,
                    "Validation_loss": val_loss,
                    })
                    
                hubert.train()
                
                ### save if there's improvement in loss or target metric ###
                
                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT: 
                    
                    best_val_loss = val_loss
                    patience_count = 0
                    checkpoint = {
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'model_config': hubert.config,
                        'model_state_dict': copy.deepcopy(hubert.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
                        'patience_count': patience_count,
                        'linear' : True if args.classifier_hidden_size is None and not args.use_label_embedding else False,
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    Path(SUPERVISED_MODEL_CKPT_PATH).mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, checkpoint_name))
                    # torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))
                    
                    logger.info(f"New best val loss = {best_val_loss}. Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None # reward for loss
                                
                elif target_score >= best_val_target_score + MINIMAL_IMPROVEMENT:
                    
                    best_val_target_score = target_score
                    
                    # do not increment patience count but do not reset it either
                    
                    checkpoint = {
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'model_config': hubert.config,
                        'model_state_dict': copy.deepcopy(hubert.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
                        'patience_count': patience_count,
                        'linear' : True if args.classifier_hidden_size is None and not args.use_label_embedding else False,
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    Path(SUPERVISED_MODEL_CKPT_PATH).mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, checkpoint_name))
                    # torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))

                    
                    logger.info(f"Val loss not improved but {args.target_metric} did (= {target_score}). Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None # reward for target metric
                    
                else: # loss not improved and target metric not improved
                    patience_count += 1
                    
                    if args.dynamic_reg and patience_count % (args.patience // args.intervals_for_penalty) == 0 and patience_count != args.patience:
                        dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=True) # penalize model with regularization but not too much
                                
                    if patience_count == args.patience:
                        logger.warning(f"Early stopping at step {global_step}.")
                        wandb.log({
                            "patience_count": patience_count
                            })
                        return
                
    logger.info("End of finetuning.")
    logger.info(f"Best val loss = {best_val_loss}")
    logger.info(f"Best val target score = {best_val_target_score}, ({args.target_metric})")


if __name__ == "__main__":
    args = create_parser()
    quit()
    # Start training
    finetune(args)
