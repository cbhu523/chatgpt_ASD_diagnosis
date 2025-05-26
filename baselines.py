#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune five Transformer baselines (BERT, RoBERTa, DistilBERT, ALBERT, XLNet)
for binary text classification on caltech_A4.json.

"""

# -------------------- 0. install / imports  --------------------
# !pip install -q transformers datasets scikit-learn evaluate

import json, random, os, torch, evaluate, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

# reproducibility
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# -------------------- 1. load data  ------------------------------------------------
DATA_PATH = "caltech_A4.json"            
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def subject_id(uid: str) -> str:
    """'140307_3' ➜ '140307'"""
    return "_".join(uid.split("_")[:2])

records = [{
    "uid"     : ex["uid"],
    "subject" : subject_id(ex["uid"]),
    "text"    : ex["text"],
    "label"   : int(ex["label"])
} for ex in raw]

ds_full = Dataset.from_list(records)

# -------------------- 2. training/testing group split ----------------------------------
gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=SEED)
train_idx, test_idx = next(gss.split(ds_full, groups=ds_full["subject"]))

dataset = DatasetDict({
    "train": ds_full.select(train_idx),
    "test" : ds_full.select(test_idx)
})
print(dataset)

# -------------------- 3. metrics -------------------------------------------------
metric_acc  = evaluate.load("accuracy")
metric_prec = evaluate.load("precision")
metric_rec  = evaluate.load("recall")
metric_f1   = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy" : metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "precision": metric_prec.compute(predictions=preds, references=labels,
                                         average="binary")["precision"],
        "recall"   : metric_rec.compute(predictions=preds, references=labels,
                                         average="binary")["recall"],
        "f1"       : metric_f1.compute(predictions=preds, references=labels,
                                         average="binary")["f1"],
    }

# -------------------- 4. baseline checkpoints --------------------------------------------
BASELINES = {
    "bert-base-uncased"      : "BERT",
    "roberta-base"           : "RoBERTa",
    "distilbert-base-uncased": "DistilBERT",
    "albert-base-v2"         : "ALBERT",
    "xlnet-base-cased"       : "XLNet"
}

# -------------------- 5. training loop -------------------------------------------------
for checkpoint, nickname in BASELINES.items():
    print(f"\n=== Fine-tuning {nickname} ({checkpoint}) ===")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenised = dataset.map(tokenize, batched=True,
                            remove_columns=["text", "uid", "subject"])

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)

    out_dir = f"./models/{checkpoint.replace('/', '-')}"
    args = TrainingArguments(
        output_dir                = out_dir,
        num_train_epochs          = 3,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        learning_rate             = 2e-5,
        evaluation_strategy       = "no",     
        save_strategy             = "epoch",
        seed                      = SEED,
        logging_steps             = 50,
        report_to                 = "none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics          
    )

    trainer.train()

    # ---------------- final test evaluation ----------------
    test_metrics = trainer.evaluate(tokenised["test"], metric_key_prefix="test")
    print(f"  Test metrics for {nickname}: {test_metrics}")

print("\nAll done!  Check './models/' for saved checkpoints.")
