from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperTokenizerFast, WhisperFeatureExtractor
import datasets
from datasets import DatasetDict, concatenate_datasets

def preprocess_datasets():
    ds1 = load_dataset('facebook/multilingual_librispeech',
                    'german',
                        cache_dir="/media/hdd/.cache/huggingface",
                        # token=token,
                        streaming=False,
                        trust_remote_code=True,
                )

    ds2 = load_dataset('mozilla-foundation/common_voice_16_0',
                        'de',
                        cache_dir="/media/hdd_old/.cache/huggingface",
                    )

    ds3 = load_dataset(
                'facebook/voxpopuli',
                'de',
                cache_dir="/media/hdd_old/.cache/huggingface",
                )

    feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-medium.en")
    tokenizer = WhisperTokenizerFast.from_pretrained("distil-whisper/distil-medium.en")

    def rename_columns(ds, column_nammes):
        ds = ds.cast_column("audio", datasets.features.Audio(16000))
    
        ds = ds.rename_column(column_nammes, "text")
        
        dataset_features = ds['train'].features.keys()
        columns_to_keep = {"audio", "text"}
        ds = ds.remove_columns(set(dataset_features - columns_to_keep))
        
        return ds
        
    ds1 = rename_columns(ds1, "transcript")
    ds2 = rename_columns(ds2, "sentence")
    ds3 = rename_columns(ds3, "raw_text")


    dataset = DatasetDict()
    dataset['train'] = concatenate_datasets([ds1['train'], ds2['train'], ds2['validation']])
    dataset['ID_eval'] = concatenate_datasets([ds1['test'], ds2['test']])
    dataset['OOD_eval'] = concatenate_datasets([ds3['validation'], ds3['test']])


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = [sample["array"] for sample in batch["audio"]]
        inputs = feature_extractor(audio, sampling_rate=16000, device='cuda')
        batch["input_features"] = inputs.input_features
        batch["input_length"] = [len(sample) for sample in audio]
        batch["labels"] = tokenizer(batch["text"]).input_ids
        
        return batch

    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['ID_eval'] = dataset['ID_eval'].shuffle(seed=42)
    dataset['OOD_eval'] = dataset['OOD_eval'].shuffle(seed=42)
    
    dataset = dataset.map(prepare_dataset, batched=True, batch_size=128, num_proc=8)
    
    max_input_length = (30 * 16000)
    min_input_length = (0 * 16000)
    max_label_length = 448  # model.config.max_length
    
    
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length
    
    dataset.filter(function=is_audio_in_length_range, input_columns=["input_length"])
    
    def is_labels_in_length_range(labels):
        return 0 < len(labels) <= max_label_length
    
    dataset.filter(function=is_labels_in_length_range, input_columns=["labels"])
    
    return dataset

if __name__ == "__main__":
    raw_dataset = preprocess_datasets()