# Whisper lora fine-tune

## Train.args:

```
base_model = "distil-whisper/distil-medium.en"
per_device_train_batch_size = 72,
per_device_eval_batch_size = 72,
gradient_accumulation_steps = 1,
learning_rate = 1e-3,
warmup_steps = 50,
num_train_epochs = 3,
eval_strategy = "steps",
fp16 = True,
generation_max_length = 128,
remove_unused_columns = False,
label_names = ["labels"],
logging_steps = 10,
eval_steps = 1000,
predict_with_generate = False,
save_steps = 1000,
dataloader_num_workers = 8,
do_train = True,

model_dtype = "float16"
attn_implementation = "flash_attention_2"
```

## Script:
```
venv/bin/accelerate config

venv/bin/accelerate launch run_lora.py
```