from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy
)
from datasets import load_dataset
import evaluate
import numpy as np

from utils import *

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(tokenizer, task, examples):
    if task == "sst2":
        outputs = tokenizer(examples[TASK_TO_COLUMNS[task][0]], truncation=True, max_length=256)
    else:
        outputs = tokenizer(examples[TASK_TO_COLUMNS[task][0]], examples[TASK_TO_COLUMNS[task][1]], truncation=True, max_length=256)
    return outputs

def run(args):
    dataset = load_dataset("glue", args.task)
    global metric
    metric = evaluate.load("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="right")
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, args.task, examples),
        batched=True,
        remove_columns=["idx"]+[column for column in TASK_TO_COLUMNS[args.task]]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(set(dataset["train"]["label"])),return_dict=True)
    trainable_parameters=count_parameters(model)
    training_args = TrainingArguments(
        output_dir=f"checkpoint/{args.training_type}/{args.model_name_or_path}/{args.task}/soft-prompts",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=50,
        save_total_limit=5,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True,
        do_train=True,
        overwrite_output_dir=True,
        remove_unused_columns=False
    )
    train_dataset=tokenized_datasets["train"]
    eval_dataset=tokenized_datasets["validation_matched" if args.task == "mnli" else "validation"]
    eval_results=0
    if args.training_type == "public":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        # Evaluate on the validation set and get the final validation metrics
        eval_results = trainer.evaluate()
    else:
        import dp_transformers
        training_args = TrainingArguments(
            output_dir=f"checkpoint/{args.training_type}/{args.model_name_or_path}/{args.task}/soft-prompts",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=0.01,
            save_total_limit=5,
            do_train=True,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            max_grad_norm=args.per_sample_max_grad_norm
        )
        private_train_args = dp_transformers.TrainingArguments(**training_args.to_dict())
        privacy_args = dp_transformers.PrivacyArguments(per_sample_max_grad_norm=0.01, target_epsilon=8, target_delta=1/len(dataset["train"]))
        model.train()
        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=private_train_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            privacy_args=privacy_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()

    return eval_results, trainable_parameters