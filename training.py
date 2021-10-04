import torch
import transformers
from torch.utils.data import Dataset
from transformers import BertTokenizer, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from sparsebert import SSAFForSequenceClassification
import random
import numpy


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class SSTDataset(Dataset):
    def __init__(self, text_dir):
        tmp_words = list()
        tmp_label = list()
        with open(text_dir) as file:
            count = 0
            for line in file:
                if count != 0 and len(line) > 0:
                    tmp1 = line.split('\t')
                    tmp_words.append(tmp1[0])
                    tmp_label.append(int(tmp1[1]))
                count = count + 1
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        self.words = tokenizer(tmp_words, padding=True, truncation=True)
        self.label = tmp_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.words.items()}
        item['labels'] = torch.tensor(self.label[idx])
        return item


set1 = SSTDataset("D:\\dataset\\SST-2\\train.tsv")
set2 = SSTDataset("D:\\dataset\\SST-2\\dev.tsv")
torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)
model = SSAFForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments("test_trainer",
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  learning_rate=2e-5,
                                  adam_epsilon=1e-8,
                                  weight_decay=0.01,
                                  warmup_ratio=0.1,
                                  num_train_epochs=4,
                                  seed=0)
training_args.evaluation_strategy = "epoch"
trainer2 = Trainer(model=model,
                   args=training_args,
                   train_dataset=set1,
                   eval_dataset=set2,
                   compute_metrics=compute_metrics)
trainer2.train()
print(trainer2.evaluate())
