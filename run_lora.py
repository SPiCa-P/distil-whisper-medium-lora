import logging

from transformers import (
WhisperForConditionalGeneration,
AdamW, 
get_scheduler, 
WhisperProcessor, 
Seq2SeqTrainingArguments,
WhisperTokenizerFast, 
WhisperFeatureExtractor,
WhisperConfig
)

from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from typing import Any, Dict, List, Optional, Union

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
import wandb
import numpy as np
from helpers import log_metric, log_pred, get_last_checkpoint

from preprocess import preprocess_datasets
from tqdm import tqdm
import time
import evaluate
import os
from multiprocess import set_start_method
import re
import copy



base_model = "distil-whisper/distil-medium.en"



logger = get_logger(__name__)


training_args = Seq2SeqTrainingArguments(
    output_dir="./ver_3",  # change to a repo name of your choice
    per_device_train_batch_size=72,
    per_device_eval_batch_size=72,
    
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    eval_strategy="steps",
    fp16=True,
    generation_max_length=128,
#    max_steps=100, # only for testing purposes, remove this from your final run :)
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    
    logging_steps=10,
    eval_steps=1000,
    predict_with_generate=False,
    save_steps=1000,
    dataloader_num_workers = 8,
    do_train=True,
)

model_dtype = "float16"
attn_implementation = "flash_attention_2"

if model_dtype == "float16":
    mixed_precision = "fp16"
    lora_dtype = torch.float16
    
elif model_dtype == "bfloat16":
    mixed_precision = "bf16"
    lora_dtype = torch.bfloat16
    
else:
    mixed_precision = "no"
    lora_dtype = torch.float32


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`)
            The start-of-prompt token id of the decoder
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        # dataloader returns a list of features which we convert to a dict
        input_features = {"input_features": [feature["input_features"] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch



class Train_lora():
    
    def __init__(self) -> None:
        # self.config = config
        self.model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype = lora_dtype,
            attn_implementation = attn_implementation,
            ).to('cuda')
        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.dataset = preprocess_datasets()
        self.tokenizer = WhisperTokenizerFast.from_pretrained(base_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
        self.config = WhisperConfig.from_pretrained(base_model)
        
    def compute_metrics(self, preds, labels):
    # replace padded labels by the padding token
        for idx in range(len(labels)):
            
            tokenizer = self.tokenizer
            return_timestamps = False
            metric = evaluate.load("wer")
            
            normalizer = (
                BasicTextNormalizer()
            )
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

            pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
            # we do not want to group tokens when computing the metrics
            label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

            # normalize everything and re-compute the WER
            norm_pred_str = [normalizer(pred) for pred in pred_str]
            norm_label_str = [normalizer(label) for label in label_str]
            # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
            pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
            # filtering step to only evaluate the samples that correspond to non-zero normalized references:
            norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
            norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

            wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
            return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str
        
    def prepare_lora_model(self):
        # define loRa configration 
        target_modules = []
        keywords = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        for id, (name, param) in enumerate(self.model.named_modules()):
            if 'model.decoder' in name and (any(keyword in name for keyword in keywords)):
                target_modules.append(name)
        
        
        lora_config = LoraConfig(
            r=8,                    # Rank param
            lora_alpha=32,           # alpha
            target_modules=target_modules,  # traget models
            lora_dropout=0.1,        # dropout prob
            bias="none",             # not using bias
        )
        
        model = get_peft_model(self.model, lora_config)
        logger.info(model.print_trainable_parameters())
        # 将 LoRA 配置应用到模型
        return model
    
    def main(self):
        
        
        accelerator = Accelerator(
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb",
            project_dir="./",
        )
        
        accelerator.init_trackers(
            project_name="distil-whisper-medium-de-lora",
            init_kwargs={
                "wandb": {"name": 'distil-whisper-medium-de-lora-ver-3',
                          "dir": "./wandb"}
            }
        )
            # 3. Set-up basic logging
        # Create one log on every process with the configuration for debugging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        

        
        # lora model
        model = self.prepare_lora_model()
        processor = self.processor
        tokenizer = self.tokenizer
        feature_extractor = self.feature_extractor
        config = self.config

        # prepare AdamW optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )

        
        train_batch_size = training_args.per_device_train_batch_size
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(self.dataset["train"]) // (train_batch_size * training_args.gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
        
        
        # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
            num_training_steps=total_train_steps * accelerator.num_processes,
        )
        
        # prepare data_collator
        
        decoder_start_token_id = model.config.decoder_start_token_id  # <|startoftranscript|>
        decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=decoder_start_token_id,
            decoder_prev_token_id=decoder_prev_token_id,
            input_padding="longest",
            target_padding="max_length",
            max_target_length=448,
        )
        
        # translation
        gen_kwargs = {
            "max_length": training_args.generation_max_length,
            "num_beams": 5,
            # "language": 'de', 
            # "task": 'transcription',
        }
        
        # Prepare everything with accelerate
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
        
        
        
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {total_train_steps * train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
        logger.info("  Gradient accumulation steps =" f" {training_args.gradient_accumulation_steps}")
        logger.info(
            f"  Total train batch size (w. parallel & distributed) = {train_batch_size * training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {total_train_steps}")
        
        # =============================================start training=====================v=====================
        
        
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        
        
        
        train_time = 0
        train_start = time.time()
        steps_trained_progress_bar = tqdm(
            range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
        )
        cur_step = 0
        epochs_trained = 0
        continue_training = True
        checkpoint = None
        eval_steps = training_args.eval_steps
        all_eval_splits = list(self.dataset.keys())
        all_eval_splits.remove("train")
        
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # print(f"last_checkpoint: {last_checkpoint}")
        print(f"Find checkpoint: {checkpoint}")
        
        
        
        if checkpoint is not None:
            accelerator.load_state(checkpoint)
            # Find num steps and epoch from saved state string pattern
            pattern = r"checkpoint-(\d+)-epoch-(\d+)"
            match = re.search(pattern, checkpoint)
            cur_step = int(match.group(1))
            epochs_trained = int(match.group(2))

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {cur_step}")

            steps_trained_progress_bar.update(cur_step)

            for epoch in range(0, epochs_trained):
                self.dataset["train"] = self.dataset["train"].shuffle(training_args.seed)

            if training_args.max_steps < 0:
                # we know exactly the number of steps per epoch, so can skip through the required number of batches
                resume_step = (cur_step - epochs_trained * steps_per_epoch) * training_args.gradient_accumulation_steps
                
            # # TODO:hot fix for this time
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = 1e-5
            #     param_group['initial_lr'] = 1e-5
            # lr_scheduler = get_scheduler(
            #     name=training_args.lr_scheduler_type,
            #     optimizer=optimizer,
            #     num_warmup_steps=0,
            #     num_training_steps= (total_train_steps - cur_step) * accelerator.num_processes,  
            # )   
                            
            
        else:
            resume_step = None
        
        
        for epoch in range(epochs_trained, training_args.num_train_epochs):
            self.dataset['train'] = self.dataset['train'].shuffle(training_args.seed)
            train_dataloader = DataLoader(
                self.dataset['train'],
                collate_fn=data_collator,
                batch_size=training_args.per_device_train_batch_size,
                num_workers=training_args.dataloader_num_workers,
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            
            if resume_step is not None:
                # Skip the first N batches in the dataloader when resuming from a checkpoint
                train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
                resume_step = None
                
                
            for batch in train_dataloader:
                
                
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # torch.save([param for param in model.parameters()], f'param{cur_step}.pt')
                # torch.save([param.grad for param in model.parameters()], f'gradients{cur_step}.pt')
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                if accelerator.sync_gradients:
                    steps_trained_progress_bar.update(1)
                    cur_step += 1
                    
                    if cur_step % training_args.logging_steps == 0:
                        steps_trained_progress_bar.write(
                            f"Epoch {epoch} | "
                            f"Step... ({cur_step} / {total_train_steps} | Loss:"
                            f" {loss}, Learning Rate:"
                            f" {lr_scheduler.get_last_lr()[0]})"
                        )
                        log_metric(
                            accelerator,
                            metrics=loss,
                            learning_rate=lr_scheduler.get_last_lr()[0],
                            train_time=train_time + time.time() - train_start,
                            step=cur_step,
                            epoch=epoch,
                            prefix="train",
                        )
                                        # save checkpoint and weights after each save_steps and at the end of training
                    if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                        intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                        accelerator.save_state(output_dir=intermediate_dir)
                        feature_extractor.save_pretrained(intermediate_dir)
                        tokenizer.save_pretrained(intermediate_dir)
                        config.save_pretrained(intermediate_dir)
                        model.save_pretrained(intermediate_dir)

                        # accelerator.wait_for_everyone()
                        # if accelerator.is_main_process:
                        #     rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)
                        
                        
                        
                    if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                        train_time += time.time() - train_start
                        model.eval()
                        wer_l, labels_l = [], []
                        
                        def generate_step(batch):
                            model.eval()
                            output_ids = accelerator.unwrap_model(model).generate(batch["input_features"], **gen_kwargs)
                            output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
                            return output_ids
                        
                        # ======================== Evaluating ==============================
                        for eval_split in all_eval_splits:
                            eval_metrics = []
                            eval_preds = []
                            eval_labels = []
                            eval_start = time.time()

                            validation_dataloader = DataLoader(
                                self.dataset[eval_split],
                                collate_fn=data_collator,
                                batch_size=training_args.per_device_eval_batch_size,
                                drop_last=False,
                                num_workers=training_args.dataloader_num_workers,
                                pin_memory=training_args.dataloader_pin_memory,
                            )
                            validation_dataloader = accelerator.prepare(validation_dataloader)
                            
                            for batch in tqdm(
                                validation_dataloader,
                                desc=f"Evaluating {eval_split}...",
                                position=2,
                                disable=not accelerator.is_local_main_process,
                            ):
                                # Model forward
                                with torch.no_grad():
                                    outputs = model(**batch)
                                eval_metric = outputs.loss
                                eval_metric = accelerator.gather_for_metrics(eval_metric)
                                eval_metrics = {'loss' : eval_metric}

                                # generation
                                if training_args.predict_with_generate:
                                    generated_ids = generate_step(batch)
                                    # Gather all predictions and targets
                                    generated_ids, labels = accelerator.gather_for_metrics(
                                        (generated_ids, batch["labels"])
                                    )
                                    eval_preds.extend(generated_ids)
                                    eval_labels.extend(labels)

                            eval_time = time.time() - eval_start
                            # normalize eval metrics
                            # eval_metrics = {
                            #     key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                            # }
                            
                            wer_desc = ""
                            if training_args.predict_with_generate:
                                wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = self.compute_metrics(
                                    eval_preds, eval_labels
                                )
                                eval_metrics.update(wer_metric)
                                wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
                                log_pred(
                                    accelerator,
                                    pred_str,
                                    label_str,
                                    norm_pred_str,
                                    norm_label_str,
                                    step=cur_step,
                                    prefix=eval_split,
                                )
                                
                                wer_l.append(wer_metric)
                                labels_l.append(norm_label_str)

                            # Print metrics and update progress bar
                            steps_trained_progress_bar.write(
                                f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                                f" {wer_desc})"
                            )


                            log_metric(
                                accelerator,
                                metrics=eval_metrics,
                                train_time=eval_time,
                                step=cur_step,
                                epoch=epoch,
                                prefix=eval_split,
                            )

                        
        accelerator.end_training()
                

if __name__ == "__main__":
    
    set_start_method("spawn")
    
    training = Train_lora()
    training.main()
