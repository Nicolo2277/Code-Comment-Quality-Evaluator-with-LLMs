import os
import logging
import re
import numpy
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from transformers import TrainerCallback
from transformers import DebertaV2ForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

import wandb

#Logging setup
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class QualityDataset:
    '''Class for the Code-Comment dataset for quality evaluation'''
    def __init__(self, json_file, tokenizer, max_tokens=1024, seed=42):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.json_file = json_file
        self.seed  = seed
    
    def load_and_prepare_data(self, test_size=0.2, val_size=0.1):
        '''Load the data and splits into train/val/test'''
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        #Filter out unlabeled examples:
        labeled_data = [item for item in data
                        if all(item[key] is not None for key in ['clarity', 'usefulness', 'accuracy', 'overall'])]
        
        logging.info(f'Total examples: {len(data)}')
        logging.info(f'Labeled examples: {len(labeled_data)}')

        df = pd.DataFrame(labeled_data)
        df = df.reset_index(drop=True)

        #Split the data
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=self.seed)
        val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=self.seed)

        logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        #Convert to HF datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        #Apply tokenization
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        # Remove all columns except the ones needed for training
        columns_to_keep = ['input_ids', 'attention_mask', 'clarity', 'usefulness', 'accuracy', 'overall']
        columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]

        train_dataset = train_dataset.remove_columns(columns_to_remove)
        val_dataset = val_dataset.remove_columns(columns_to_remove) 
        test_dataset = test_dataset.remove_columns(columns_to_remove)

        return train_dataset, val_dataset, test_dataset
    
    def tokenize_function(self, examples):
        '''Tokenization function for the dataset'''
        #Format input texts
        inputs = [
            f'Rate this comment quality for Code: {code} [SEP] Comment: {comment}' for code, comment in zip(examples['code'], examples['comment'])
        ]
        #Tokenize
        tokenized = self.tokenizer(
            inputs, 
            truncation=True,
            padding='max_length',
            max_length=self.max_tokens,
            return_tensors=None
        )
        #Add labels
        tokenized['clarity'] = [float(x) for x in examples['clarity']]
        tokenized['usefulness'] = [float(x) for x in examples['usefulness']]
        tokenized['accuracy'] = [float(x) for x in examples['accuracy']]
        tokenized['overall'] = [float(x) for x in examples['overall']]
        
        return tokenized


class EvaluationCallback(TrainerCallback):
    """Custom callback to handle evaluation logging"""
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called after evaluation"""
        logger.info(f"Evaluation completed at step {state.global_step}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced logging callback"""
        if logs is None:
            return
            
        # Log evaluation metrics to wandb
        if hasattr(args, 'report_to') and 'wandb' in args.report_to:
            wandb_logs = {}
            
            # Handle evaluation logs
            if 'eval_loss' in logs:
                wandb_logs['eval/loss'] = logs['eval_loss']
                logger.info(f"Validation Loss: {logs['eval_loss']:.4f}")
            
            # Handle training logs
            if 'train_loss' in logs:
                wandb_logs['train/epoch_loss'] = logs['train_loss']
                logger.info(f"Training Loss: {logs['train_loss']:.4f}")
            
            if 'learning_rate' in logs:
                wandb_logs['train/learning_rate'] = logs['learning_rate']
            
            if 'epoch' in logs:
                wandb_logs['train/epoch'] = logs['epoch']
                
            # Log epoch summary
            if 'epoch' in logs:
                logger.info(f"=== EPOCH {logs['epoch']:.0f} SUMMARY ===")
                if 'train_loss' in logs:
                    logger.info(f"Training Loss: {logs['train_loss']:.4f}")
                if 'eval_loss' in logs:
                    logger.info(f"Validation Loss: {logs['eval_loss']:.4f}")
                if 'learning_rate' in logs:
                    logger.info(f"Learning Rate: {logs['learning_rate']:.2e}")
                logger.info("=" * 30)
            
            if wandb_logs:
                wandb.log(wandb_logs, step=state.global_step)


class MultiTaskTrainer(Trainer):
    '''Custom trainer for multiclass regression'''

    def _remove_unused_columns(self, dataset, description=None):
        """Override to keep our label columns"""
        # Define columns we want to keep
        columns_to_keep = ['input_ids', 'attention_mask', 'clarity', 'usefulness', 'accuracy', 'overall']
        
        # Only remove columns that aren't in our keep list
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        
        return dataset
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to handle custom labels"""
        # Extract labels for loss computation but don't pass to model
        labels = {}
        for key in ['clarity', 'usefulness', 'accuracy', 'overall']:
            if key in inputs:
                labels[key] = inputs.pop(key)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            
        if prediction_loss_only:
            # Compute loss if needed
            if labels:
                logits = outputs.logits
                loss_clarity = F.mse_loss(logits[:, 0], labels['clarity'].float())
                loss_usefulness = F.mse_loss(logits[:, 1], labels['usefulness'].float())
                loss_accuracy = F.mse_loss(logits[:, 2], labels['accuracy'].float())
                loss_overall = F.mse_loss(logits[:, 3], labels['overall'].float())
                total_loss = loss_clarity + loss_usefulness + loss_accuracy + loss_overall
                return (total_loss, None, None)
            else:
                return (outputs.loss, None, None)
        else:
            return (outputs.loss, outputs.logits, labels if labels else None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''Custom MSE loss function'''
        labels_clarity = inputs.pop('clarity').float()
        labels_usefulness = inputs.pop('usefulness').float()
        labels_accuracy = inputs.pop('accuracy').float()
        labels_overall = inputs.pop('overall').float()

        outputs = model(**inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            loss_clarity = F.mse_loss(logits[:, 0], labels_clarity)
            loss_usefulness = F.mse_loss(logits[:, 1], labels_usefulness)
            loss_accuracy = F.mse_loss(logits[:, 2], labels_accuracy)
            loss_overall = F.mse_loss(logits[:, 3], labels_overall)

            # Log individual predictions vs labels for debugging
            if self.state.global_step % 50 == 0:  # Log every 50 steps
                logger.info(f"Step {self.state.global_step} - Sample predictions vs labels:")
                logger.info(f"  Clarity: pred={logits[0, 0].item():.2f}, true={labels_clarity[0].item():.2f}")
                logger.info(f"  Usefulness: pred={logits[0, 1].item():.2f}, true={labels_usefulness[0].item():.2f}")
                logger.info(f"  Accuracy: pred={logits[0, 2].item():.2f}, true={labels_accuracy[0].item():.2f}")
                logger.info(f"  Overall: pred={logits[0, 3].item():.2f}, true={labels_overall[0].item():.2f}")
        
        total_loss = loss_clarity + loss_usefulness + loss_accuracy + loss_overall

        # Log step-level metrics to wandb
        if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
            wandb.log({
                'train/loss_clarity': loss_clarity.item(),
                'train/loss_usefulness': loss_usefulness.item(),
                'train/loss_accuracy': loss_accuracy.item(),
                'train/loss_overall': loss_overall.item(),
                'train/total_loss': total_loss.item(),
            }, step=self.state.global_step)
        
        return (total_loss, outputs) if return_outputs else total_loss


class ModelTrainer:
    '''Handles LoRA setup and model training'''
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.config = self._get_lora_config()

    def _get_lora_config(self):
        return {
            'lora_r': self.args.lora_r,
            'lora_alpha': self.args.lora_alpha,
            'lora_dropout': self.args.lora_dropout,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size_lora,
            'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
            'num_epochs': self.args.lora_epochs,
            'warmup_ratio': self.args.warmup_ratio_lora,
            'weight_decay': self.args.lora_weight_decay,
            'max_grad_norm': 1.0
        }

    def apply_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, #Sequence classification task
            inference_mode=False,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=["query_proj", "value_proj", 'key_proj'], #target the attention layers
            bias='none'
        )
        #apply lora:.
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

    def create_trainer(self, train_dataset, val_dataset, output_dir, use_wandb=False):

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            num_train_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            max_grad_norm=self.config['max_grad_norm'],

            #Logging and evaluation:
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_first_step=True,

            #optimization settings:
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            remove_unused_columns=False, 

            #Checkpoints:
            save_total_limit=1,
            save_only_model=True,

            report_to='wandb' if use_wandb else 'none',
            run_name=f"LoRA-r{self.config['lora_r']}" if use_wandb else None
        )
        
        # Create callbacks
        callbacks = [EvaluationCallback()]
        
        trainer = MultiTaskTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
        )
        
        # Set label names to suppress warning
        trainer.label_names = ['clarity', 'usefulness', 'accuracy', 'overall']
        
        return trainer


def main():

    args = cfg.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", 
        num_labels=4  # clarity, usefulness, accuracy, overall
    )
    trainer_obj = ModelTrainer(model=model, args=args)

    model = trainer_obj.apply_lora()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    processor = QualityDataset(args.data_file, tokenizer, max_tokens=args.max_tokens, seed=args.seed)

    train_dataset, val_dataset, test_dataset = processor.load_and_prepare_data(
        test_size=args.test_size,
        val_size=args.val_size
        )
    
    logger.info('Dataset loaded successfully')

    #Initialize wandb:
    if args.use_wandb:
        run_name = f"LoRA-r{trainer_obj.config['lora_r']}"
        wandb.init(
            project='LoRA fine tuning for quality comment evaluation', 
            name=run_name,
            config={
                **trainer_obj.config,
                'model_name': 'microsoft/deberta-v3-base',
                'max_tokens': args.max_tokens,
                'test_size': args.test_size,
                'val_size': args.val_size
            }
        )
    
    #Create trainer
    trainer = trainer_obj.create_trainer(train_dataset, val_dataset, output_dir=args.output_dir, use_wandb=args.use_wandb)
    logger.info('Start training')
    train_result = trainer.train()
    logger.info(f'Training complete, final loss: {train_result.training_loss}')

    #save model
    logger.info('Saving the model')
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    #Evaluate on the test set
    if test_dataset:
        logger.info('Evaluation on test set')
        test_results = trainer.evaluate(test_dataset)
        logger.info(f'Test results: {test_results}')
        
        # Log test results to wandb
        if args.use_wandb:
            wandb.log({
                'test/loss': test_results.get('eval_loss', 0),
                'test/final': True
            })

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()