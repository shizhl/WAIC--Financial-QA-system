import logging
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from args import DataTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from datasets import load_dataset,DownloadConfig

logger = logging.getLogger(__name__)


class DatasetMaker:
    def __init__(self, dataset_saved_path: str, data_args: DataTrainingArguments,
                 training_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizerBase):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.dataset_saved_path = dataset_saved_path

    def make_dataset(self):
        logger.info('******* Making Dataset **********')
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file
            extension = self.data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))
        # datasets = load_dataset(extension, data_files=data_files)
        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0:
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for model. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples: Dict):
            """
            如果是json，examples就是json对应的dict。如果是纯文本，examples["text"]就是全部文本,每个item就是文本文件中的一行
            """
            input_key = 'content'
            output_key = 'summary'
            if isinstance(examples[input_key][0], str):
                inputs = [ex.replace(' ', '') if self.data_args.chinese_data else ex for ex in examples[input_key]]
            elif isinstance(examples[input_key][0], list):
                inputs = [' '.join(ex).replace(' ', '') if self.data_args.chinese_data else ' '.join(ex) for ex in
                          examples[input_key]]
            else:
                raise ValueError(f'only support str/list in {input_key}, now {type(examples[input_key][0])}')

            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = self.tokenizer.sep_token
                # self.tokenizer.eos_token_id = self.tokenizer.cls_token_id
            if isinstance(examples[output_key][0], str):
                targets = [ex.replace(' ',
                                      '') + self.tokenizer.eos_token if self.data_args.chinese_data else ex + self.tokenizer.eos_token
                           for ex in examples[output_key]]
            elif isinstance(examples[output_key][0], list):
                targets = [' '.join(ex).replace(' ',
                                                '') + self.tokenizer.eos_token if self.data_args.chinese_data else ' '.join
                                                                                                                   (ex) + self.tokenizer.eos_token
                           for ex in examples[output_key]]
            else:
                raise ValueError(f'only support str/list in {output_key}, now {type(examples[output_key][0])}')

            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                          truncation=True,
                                          add_special_tokens=False)
            # addi_source = tokenizer(addi_source, max_length=data_args.max_source_length, padding=False, truncation=True,
            #                         add_special_tokens=False)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,
                                        add_special_tokens=False)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            # model_inputs["addi_source"] = addi_source["input_ids"]
            # model_inputs["addi_source_attention_mask"] = addi_source["attention_mask"]
            return model_inputs

        datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        logger.info('saving dataset')
        dataset_saved_path = self.dataset_saved_path
        datasets.save_to_disk(dataset_saved_path)
        logger.info(f'******* Dataset Finish {dataset_saved_path} **********')
        return datasets