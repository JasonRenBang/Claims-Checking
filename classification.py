import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import BertTokenizer,BertModel
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.data import Dataset
from torch import nn
import wandb

class TrainDataset(Dataset):
    def __init__(self,dataset, evidenceDataset, labelids, tokenizer, max_length):
        self.max_length = max_length
        self.dataset = dataset
        self.evidences = evidenceDataset
        self.label2ids = labelids
        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [(data["claim_text"]).lower()]
        for evidence_id in data["evidences"]:
            input_text.append(self.evidences[evidence_id].lower())
        input_text = self.tokenizer.sep_token.join(input_text)

        label = self.label2ids[data["claim_label"]]

        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        labels = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)

            labels.append(label)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding

class ValDataset(Dataset):
    def __init__(self,dataset,evidencedata, label2ids, tokenizer, max_length):
        self.max_length = max_length
        self.dataset = dataset 
        self.evidences = evidencedata
        self.label2ids = label2ids
        self.tokenizer = tokenizer

        self.claim_ids = list(self.dataset.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [(data["claim_text"]).lower()]
        for evidence_id in data["evidences"]:
            input_text.append(self.evidences[evidence_id].lower())
        input_text = self.tokenizer.sep_token.join(input_text)

        label = self.label2ids[data["claim_label"]]

        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        labels = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)

            labels.append(label)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding

class TestDataset(Dataset):
    def __init__(self,  label2ids, tok, max_length):
        self.max_length = max_length

        f = open("retrieval-test-claims.json", "r")
        self.dataset = json.load(f)
        f.close()
        f = open("evidence.json", "r")
        self.evidences = json.load(f)
        f.close()

        self.label2ids = label2ids
        self.tokenizer = tok
        self.claim_ids = list(self.dataset.keys())
        

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [data["claim_text"].lower()]
        for evidence_id in data["evidences"]:
            input_text.append(self.evidences[evidence_id].lower())
        input_text = self.tokenizer.sep_token.join(input_text)

        label = None
        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        return batch_encoding
    
class TestDataset(Dataset):
    def __init__(self,dataset,evidencedataset,  label2ids, tok, max_length):
        self.max_length = max_length

        self.dataset = dataset
        self.evidences = evidencedataset
 
        self.label2ids = label2ids
        self.tokenizer = tok
        self.claim_ids = list(self.dataset.keys())
        

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [data["claim_text"].lower()]
        for evidence_id in data["evidences"]:
            input_text.append(self.evidences[evidence_id].lower())
        input_text = self.tokenizer.sep_token.join(input_text)

        label = None
        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        return batch_encoding


class CLSModel(nn.Module):
    def __init__(self, pre_encoder):
        super(CLSModel, self).__init__()
        self.encoder =BertModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.cls = nn.Linear(hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        texts_emb = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)
        return logits
    

wandb.init(project="nlp assignment 3", name="classification")
train = "train-claims.json"
with open (train, "r") as json_file:
    train_dataset = json.load(json_file)

evidence = "evidence.json"
with open (evidence, "r") as json_file:
    evidence_dataset = json.load(json_file)

val = "dev-claims.json"
with open (val, "r") as json_file:
    val_dataset = json.load(json_file)

test = "retrieval-predictions.json"
with open (val, "r") as json_file:
    test_dataset = json.load(json_file)

claims = []
for key in test_dataset:
    claims.append(test_dataset[key]["claim_text"])
print(len(claims))
longestLengthIndex, longestLength = max(enumerate(claims), key=lambda x: len(x[1]))
print(longestLengthIndex)
print(longestLength)
print(len(longestLength))
shortestLengthIndex, shortestLength = min(enumerate(claims), key=lambda x: len(x[1]))
print(shortestLengthIndex)
print(shortestLength)
print(len(shortestLength))



torch.manual_seed(600)
torch.cuda.manual_seed_all(600)
np.random.seed(600)
random.seed(600)

tokenizer =BertTokenizer.from_pretrained('bert-base-cased')

label_ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
train_set = TrainDataset(train_dataset,evidence_dataset,label_ids, tokenizer, 256)
val_set = ValDataset(val_dataset,evidence_dataset,label_ids, tokenizer, 256)

train_dataloader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,

    collate_fn=train_set.collate_fn,
)
val_dataloader = DataLoader(
    val_set, batch_size=8, shuffle=False, collate_fn=val_set.collate_fn
)

model = CLSModel("bert-base-cased")
model.cuda()
model.train()

lossFunction = nn.CrossEntropyLoss()
s_optimizer = optim.Adam(model.parameters())

for param_group in s_optimizer.param_groups:
    param_group["lr"] = 5e-5

# start training
s_optimizer.zero_grad()
step_cnt = 0
all_step_cnt = 0
averageLossValue = 0
maxAccuracy = 0


for epoch in range(15):
    epoch_step = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        for n in batch.keys():
            if n in ["input_ids", "attention_mask", "label"]:
                batch[n] = batch[n].cuda()
        step_cnt += 1
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = lossFunction(logits, batch["label"])
        loss = loss / 4
        loss.backward()

        averageLossValue += loss.item()
        if step_cnt == 4:
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            step_cnt = 0
            epoch_step += 1
            all_step_cnt += 1
            if all_step_cnt <= 150:
                lr = all_step_cnt * (5e-5 - 5e-8) / 150 + 5e-8
            else:
                lr = 5e-5 - (all_step_cnt - 150) * 1e-8
            for param_group in s_optimizer.param_groups:
                param_group["lr"] = lr
            s_optimizer.step()
            s_optimizer.zero_grad()
        if all_step_cnt % 10 == 0 and step_cnt == 0:
            if all_step_cnt <= 150:
                lr = all_step_cnt * (5e-5 - 5e-8) / 150 + 5e-8
            else:
                lr = 5e-5 - (all_step_cnt - 150) * 1e-8
            wandb.log({"learning_rate": lr}, step=all_step_cnt)
            wandb.log({"loss": averageLossValue / 10 }, step=all_step_cnt)


            averageLossValue = 0
        del loss, logits
        if all_step_cnt % 20 == 0 and all_step_cnt != 0 and step_cnt == 0:

  
            model.eval()
            cnt = 0.0
            correct_cnt = 0.0
            for batch in tqdm(val_dataloader):
                for n in batch.keys():
                    if n in ["input_ids", "attention_mask", "label"]:
                        batch[n] = batch[n].cuda()
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                predictLabels = logits.argmax(-1)
                result = predictLabels == batch["label"]
                correct_cnt += result.sum().item()
                cnt += predictLabels.size(0)
            acc = correct_cnt / cnt

            wandb.log({"accuracy": acc}, step=all_step_cnt)

            if acc > maxAccuracy:
                maxAccuracy = acc


ids2label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
test_set = TestDataset(train_dataset,evidence_dataset,label_ids, tokenizer, 256)
test_dataloader = DataLoader(
    test_set,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=test_set.collate_fn,
)

result = {}
for batch in tqdm(test_dataloader):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].cuda()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    predictLabels = logits.argmax(-1).tolist()
    idx = 0
    for data, predict_label in zip(batch["datas"], predictLabels):
        data["claim_label"] = ids2label[predict_label]
        result[batch["claim_ids"][idx]] = data
        idx += 1
fout = open("test-claims-predictions.json", "w")
json.dump(result, fout)
fout.close()