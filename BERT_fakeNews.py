

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import torch.optim as optim
import gc #garbage collector for gpu memory 
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
news_data = pd.read_csv("news.csv",header=0)
def label_target(row):
    if row["label"] == "FAKE":
        return 0
    else:
        return 1
news_data["target"] = news_data.apply(lambda row: label_target(row), axis=1)
news_data.columns = ["id","title","text","target_names","target"]
del news_data["id"]
del news_data["title"]


# #### The transformers package comes with a tokenizer for each model. We"ll use the BERT tokenizer here and a BERT base model where the text isn"t modified for case.

# In[7]:


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokenized_df = list(map(lambda t: ["[CLS]"] + tokenizer.tokenize(t)[:510] + ["[SEP]"], news_data["text"]))


totalpadlength = 512



indexed_tokens = list(map(tokenizer.convert_tokens_to_ids, tokenized_df))




index_padded = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in indexed_tokens])



target_variable = news_data["target"].values




all_words = []
for l in tokenized_df:
  all_words.extend(l)
all_indices = []
for i in indexed_tokens:
  all_indices.extend(i)

word_to_ix = dict(zip(all_words, all_indices))
ix_to_word = dict(zip(all_indices, all_words))


mask_variable = [[float(i>0) for i in ii] for ii in index_padded]


BATCH_SIZE = 14
def format_tensors(text_data, mask, labels, batch_size):
    X = torch.from_numpy(text_data)
    X = X.long()
    mask = torch.tensor(mask)
    y = torch.from_numpy(labels)
    y = y.long()
    tensordata = data_utils.TensorDataset(X, mask, y)
    loader = data_utils.DataLoader(tensordata, batch_size=batch_size, shuffle=False)
    return loader

X_train, X_test, y_train, y_test = train_test_split(index_padded, target_variable, 
                                                    test_size=0.1, random_state=42)

train_masks, test_masks, _, _ = train_test_split(mask_variable, index_padded, 
                                                       test_size=0.1, random_state=42)

trainloader = format_tensors(X_train, train_masks, y_train,BATCH_SIZE)
testloader = format_tensors(X_test, test_masks, y_test, BATCH_SIZE)





next(iter(trainloader))



model = BertForSequenceClassification.from_pretrained("bert-base-cased")

# #### Creating a function to compute the accuracy after each epoch

# In[18]:


def compute_accuracy(model, dataloader, device):
    tqdm()
    model.eval()
    correct_preds, num_samples = 0,0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            token_ids, masks, labels = tuple(t.to(device) for t in batch)
            _, yhat = model(input_ids=token_ids, attention_mask=masks, labels=labels)
            prediction = (torch.sigmoid(yhat[:,1]) > 0.5).long()
            num_samples += labels.size(0)
            correct_preds += (prediction==labels.long()).sum()
            del token_ids, masks, labels #memory
        torch.cuda.empty_cache() #memory
        gc.collect() # memory
        return correct_preds.float()/num_samples*100


# #### Now we iterate through the dataset, updating the model weights at each instance. Since BERT is pre-trained, we keep the learning rate low and only perform a few epochs. This prevents it from overfitting.

# In[20]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() #memory
gc.collect() #memory
NUM_EPOCHS = 3
loss_function = nn.BCEWithLogitsLoss()
losses = []
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-6)
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    iteration = 0
    for i, batch in enumerate(trainloader):
        iteration += 1
        token_ids, masks, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        loss, yhat = model(input_ids=token_ids, attention_mask=masks, labels=labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        del token_ids, masks, labels #memory
    
        if not i%25:
            print(f"Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} |") 
            running_loss = 0.0
            iteration = 0
        torch.cuda.empty_cache() #memory
        gc.collect() #memory
        losses.append(float(loss.item()))
        


# #### Finally, we score the final model on the test set

# In[ ]:

# #### We then do some error analysis by gathering the articles that were incorrectly predicted and analyzing the text of the articles.

# In[ ]:


test_predictions = torch.zeros((len(y_test),1))
test_predictions_percent = torch.zeros((len(y_test),1))
with torch.no_grad():
  for i, batch in enumerate(tqdm(testloader)):
    token_ids, masks, labels = tuple(t.to(device) for t in batch)
    _, yhat = model(input_ids=token_ids, attention_mask=masks, labels=labels)
    prediction = (torch.sigmoid(yhat[:,1]) > 0.5).long().view(-1,1)
    test_predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = prediction
    test_predictions_percent[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.sigmoid(yhat[:,1]).view(-1,1)


# In[ ]:


X_train_words, X_test_words, y_train_words, y_test_words = train_test_split(news_data["text"], target_variable, 
                                                    test_size=0.1, random_state=42)


# In[ ]:


final_results = X_test_words.to_frame().reset_index(drop=True)
final_results["predicted"] = np.array(test_predictions.reshape(-1), dtype=int).tolist()
final_results["percent"] = np.array(test_predictions_percent.reshape(-1), dtype=float).tolist()
final_results["actual"] = y_test_words
wrong_results = final_results.loc[final_results["predicted"]!=final_results["actual"]].copy()


# In[ ]:


print("Number of incorrectly classified articles:", len(wrong_results))


# #### This displays the incorrectly predicted instances, along with the percent confidence the algorithm had in each instance. The threshold for classification is 50%. Instances closer to 100% are more confident it"s real news and instances closer to 0% are more confident it"s fake news.

# In[ ]:
