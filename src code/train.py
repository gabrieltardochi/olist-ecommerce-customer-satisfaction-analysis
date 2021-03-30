import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn import metrics
from transformers import AutoModel, BertTokenizerFast, AdamW


class ReviewsDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = self.review[item]
        target = self.target[item]
        
        return review, target


class BERTPortugueseSentClass(nn.Module):
    def __init__(self, bert_pretrained):
        super(BERTPortugueseSentClass, self).__init__()
        self.bert = bert_pretrained
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        _, x = self.bert(ids, mask)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        output = self.sigmoid(x)
        return output


if __name__ == "__main__":
    torch.cuda.empty_cache()

    df = pd.read_csv("../data/processed/order_review_classification_ptbr.csv", sep=";", keep_default_na=False, na_values=[''])
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "pos" else 0)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.2, random_state=0, stratify=df.sentiment.values, shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = ReviewsDataset(
        review=df_train.comments.values, target=df_train.sentiment.values
    )

    valid_dataset = ReviewsDataset(
        review=df_valid.comments.values, target=df_valid.sentiment.values
    )

    BATCH_SIZE = 32

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
    bert_pretrained = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased', return_dict=False)
    
    # freeze all the bert parameters
    for param in bert_pretrained.parameters():
        param.requires_grad = False

    mdl = BERTPortugueseSentClass(bert_pretrained)

    mdl.to(device)  # gpu if available

    criterion = nn.BCELoss()
    optimizer = optimizer = AdamW(mdl.parameters(), lr = 1e-4)

    EPOCHS = 20

    min_val_loss = None
    best_mdl = None
    best_acc = None
    len_trainset = len(train_dataset)
    len_validset = len(valid_loader)
    early_stop_count = 0
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print("\nEPOCH:", epoch)
        train_running_loss = 0
        val_running_loss = 0
        mdl.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            review, label = batch
            tokens = tokenizer.batch_encode_plus(
                list(review),
                max_length = 100,
                padding='max_length',
                truncation=True
            )
            ids = torch.tensor(tokens['input_ids'], dtype=torch.long).to(device)
            mask = torch.tensor(tokens['attention_mask'], dtype=torch.long).to(device)
            targets = torch.tensor(list(label), dtype=torch.float).to(device).view(-1, 1)
            
            outputs = mdl(ids, mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        train_avg_loss = train_running_loss/len_trainset

        with torch.no_grad():
            acc_val = 0
            count_batches = 0
            mdl.eval()
            val_running_loss = 0
            for i, batch in enumerate(valid_loader):
                count_batches += 1
                
                review, label = batch
                tokens = tokenizer.batch_encode_plus(
                    list(review),
                    max_length = 100,
                    padding='max_length',
                    truncation=True
                )
                ids = torch.tensor(tokens['input_ids'], dtype=torch.long).to(device)
                mask = torch.tensor(tokens['attention_mask'], dtype=torch.long).to(device)
                targets = torch.tensor(list(label), dtype=torch.float).to(device).view(-1, 1)

                outputs = mdl(ids, mask)
                val_loss = criterion(outputs, targets).item()
                val_running_loss += val_loss
                
                y_hat = (outputs.cpu().detach().numpy().flatten() >= 0.5).astype(float)
                y_true = targets.cpu().detach().numpy().flatten()
                acc_val += (y_hat == y_true).sum()
                

        val_avg_loss = val_running_loss/len_validset
        val_acc = acc_val / (count_batches*BATCH_SIZE)
        print(f"  Train loss: {train_avg_loss}\n  Valid loss: {val_avg_loss}\n  Valid accuracy: {val_acc}")
        if min_val_loss == None or val_avg_loss < min_val_loss:
            min_val_loss = val_avg_loss
            best_mdl = mdl.state_dict()
            best_acc = val_acc
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count == 5:
            print("Early stopping on epoch", epoch)
            break

    print(f'Finished training, saving best model! Best acc = {best_acc}, minimum loss = {min_val_loss}')
    torch.save(best_mdl, f'../models/{best_acc}acc_{min_val_loss}loss.pth')

    # get predictions for all data
    mdl.load_state_dict(torch.load(f'../models/{best_acc}acc_{min_val_loss}loss.pth'))
    mdl.eval()
    preds_list = []
    preds_list = []
    with torch.no_grad():
        for i, row in df.iterrows(): 
            review = [row['comments']]
            tokens = tokenizer.batch_encode_plus(
                review,
                max_length = 100,
                padding='max_length',
                truncation=True
            )
            ids = torch.tensor(tokens['input_ids'], dtype=torch.long).to(device)
            mask = torch.tensor(tokens['attention_mask'], dtype=torch.long).to(device)
            preds = mdl(ids, mask)
            
            preds = preds.cpu().detach().numpy().flatten()
            preds_list.append(preds[0])
    df['mdl_preds'] = preds_list
    df.to_csv('../data/processed/order_reviews_sentiment_analysis.csv', index=False)