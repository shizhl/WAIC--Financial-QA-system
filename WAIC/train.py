"""
author:  shizhl
filename：模型文件（用于fine-tune中文bart）
method: 首先将处理好的数据处理为huggingface，之后利用下游任务的数据微调pre-train模型
details:
"""
import logging
import sys
import traceback
from dataset_maker import DatasetMaker
from evaluation import *
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import os
from magic_bart2 import MyBart, AutoDecodeCallback, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer
import nltk  # 用于分词
from datasets import load_dataset, DatasetDict, DownloadConfig, Features, Value

import transformers
from filelock import FileLock
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed, BartConfig
)
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bart.tokenization_bart import BartTokenizer
from args import ModelArguments, DataTrainingArguments
from datasets import load_dataset
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def main():
    # 参数选项参见args.py文件
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) 
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  
    training_args.logging_steps = 10
    data_args.log_root = os.path.join(data_args.log_root, data_args.proj_name, data_args.exp_name)
    training_args.output_dir = os.path.join(data_args.log_root, 'model')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Dataset parameters %s", data_args)

    set_seed(training_args.seed)


    if model_args.model_name_or_path is None:
        logger.info('no specific model')
        exit()
    else:
        logger.info(f'******* Loading model form pretrained {model_args.model_name_or_path} **********')
        tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path)  # 使用fnlp-bart-base/tokenizer作为基座
        model = MyBart.from_pretrained(model_args.model_name_or_path) # 使用fnlp-bart-base/tokenizer作为基座

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.save_dataset_path is None:
        maker = DatasetMaker('./datasets/', data_args, training_args, tokenizer)  # 自行指定数据处理后存放的文件夹
        datasets = maker.make_dataset()
        exit()     # 第一次处理后直接退出
    else:
        logger.info(f'******* Loading Dataset from {data_args.save_dataset_path} **********')
        datasets = DatasetDict.load_from_disk(data_args.save_dataset_path)

    train_dataset = datasets["train"] if training_args.do_train is not None and "train" in datasets else None
    eval_dataset = datasets["validation"] if training_args.do_eval is not None and "validation" in datasets else None
    test_dataset = datasets["test"] if training_args.do_predict is not None and "test" in datasets else datasets["validation"]
    if training_args.do_predict is None and "test" not in datasets:
        logging.warning(f'using validation dataset as test!')

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    max_target_length = data_args.val_max_target_length
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = MyDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    model.config.num_beams = data_args.num_beams
    model.config.max_length = data_args.max_target_length

    # Initialize Trainer
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[]  # auto_decode_callback
    )

    # Training
    if training_args.do_train:
        try:
            train_result = trainer.train()
            # train_result = trainer.train(resume_from_checkpoint=model_args.model_name_or_path) # 在线跑模型断网时，可以通过此处代码继续训练
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
        except KeyboardInterrupt:
            logger.info('stop training')
        finally:      # 键盘中断后自动保存模型
            traceback.print_exc()
            if trainer.is_world_process_zero():
                logger.info('exit, saving model')
                trainer.save_model(output_dir=os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}')) 
                trainer.state.save_to_json(os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}', 'trainer_state.json'))
            exit(0)

    # Evaluation
    results = {}
    if training_args.do_eval:   # 最后写入测试集
        if trainer.state.global_step == 0:
            trainer.state = trainer.state.load_from_json(
                os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        logger.info(f"*** Evaluate step {trainer.state.global_step} ***")

        eval_results = trainer.predict(
            test_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )

        print(eval_results.metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                eval_results.label_ids[eval_results.label_ids < 0] = tokenizer.pad_token_id
                eval_label = tokenizer.batch_decode(
                    eval_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                eval_preds = [pred.strip() for pred in eval_preds]
                eval_label = [label.strip() for label in eval_label]
                for pred, lab in zip(eval_preds[:10], eval_label[:10]):
                    logger.info(f'{pred}\t{lab}')

                dec_dir = os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}')
                print(eval_all(eval_preds,eval_label))
                if not os.path.exists(dec_dir):
                    os.makedirs(dec_dir)
                fo_ref = open(os.path.join(dec_dir, 'reference.txt'), 'w', encoding='utf8')
                fo_dec = open(os.path.join(dec_dir, 'decoded.txt'), 'w', encoding='utf8')
                for pred, lab in zip(eval_preds, eval_label):
                    fo_ref.write(f'{lab}\n')
                    fo_dec.write(f'{pred}\n')

    if training_args.do_predict:  # 最后写入测试集
        # 在训练过程中每隔一定的步数就会写入模型,此处评估的是最终训练完的模型
        logger.info(f"*** Test ***")
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        print(test_results.metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                for pred  in test_preds[:10]:
                    logger.info(f'{pred}')

                dec_dir = os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}')
                if not os.path.exists(dec_dir):
                    os.makedirs(dec_dir)
                fo_dec = open(os.path.join(dec_dir, 'decoded.txt'), 'w', encoding='utf8')
                fo_dec.write("candidates")
                for pred in test_preds:
                    pred.replace(',','，')  # 保证提价的csv文件中没有英文逗号，也可以手动通过记事本进行替换
                    fo_dec.write(f'{pred}\n')

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'  # 指定设备
    main()
