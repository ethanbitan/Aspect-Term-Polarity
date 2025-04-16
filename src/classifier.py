from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class AspectTermDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                label, aspect, term, offset, sentence = line.strip().split("\t")
                label_id = LABEL2ID[label]

                input_text = sentence
                second_segment = f"{term} {aspect}"

                tokenized = tokenizer(
                    input_text,
                    second_segment,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                self.samples.append((tokenized, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, label = self.samples[idx]
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label)
        return item


class Classifier:
    def __init__(self, ollama_url=None):
        self.model_name = "microsoft/deberta-v3-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )

    def train(self, train_filename, dev_filename, device):
        self.model.to(device)

        train_dataset = AspectTermDataset(train_filename, self.tokenizer)
        eval_dataset = AspectTermDataset(dev_filename, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_dir="./logs",
            weight_decay=0.01,
            fp16=True if torch.cuda.is_available() else False,
            logging_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

    def predict(self, data_filename, device):
        self.model.to(device)
        self.model.eval()

        dataset = AspectTermDataset(data_filename, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        preds = []
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).tolist()
                preds.extend([ID2LABEL[i] for i in batch_preds])

        return preds