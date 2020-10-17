#!/usr/bin/env python
# coding: utf-8

# # **Using a BERT Model to Predict Fake News**

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import torch.optim as optim
import gc #garbage collector for gpu memory 
from tqdm import tqdm


# #### The BERT package (transformers) has to be installed and run

# In[3]:


get_ipython().run_cell_magic('capture', '', '!pip install transformers')


# #### Import the library specific to running BERT models on PyTorch. The transformers package using the existing PyTorch infrastructure to recreate the BERT model architecture.

# In[4]:


get_ipython().run_cell_magic('capture', '', 'from transformers import BertForSequenceClassification, BertTokenizer\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')


# #### Read in the news data through the csv file. The following columns are not relevant for this endeavor:
# 
# *   ID - this is meaningless and could cause overfitting
# *   Title - for this experiment we'll choose to omit it
# 
# 
# 

# In[5]:


import pandas as pd
news_data = pd.read_csv("news.csv",header=0)
def label_target(row):
    if row["label"] == "FAKE":
        return 0
    else:
        return 1
news_data["target"] = news_data.apply(lambda row: label_target(row), axis=1)
news_data.columns = ['id','title','text','target_names','target']
del news_data['id']
del news_data['title']


# #### This is a preview of the data once the irrelevant columns have been removed. 

# In[6]:


news_data.head(10)


# #### The transformers package comes with a tokenizer for each model. We'll use the BERT tokenizer here and a BERT base model where the text isn't modified for case.

# In[7]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# #### Tokenizing the data so that each sentence is split into words and symbols. Also '[CLS]' and '[SEP]' to the beginning and end of every article.

# In[8]:


tokenized_df = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], news_data['text']))


# #### The max input length for a BERT algorithm is 512, so we'll have to pad each article to this length or cut it short.

# In[9]:


totalpadlength = 512


# #### We need to get the index for each token so that we can map them to be put in a matrix embedding.

# In[10]:


indexed_tokens = list(map(tokenizer.convert_tokens_to_ids, tokenized_df))


# In[11]:


index_padded = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in indexed_tokens])


# #### Setting up an array with the binary target variable values
# * 0 = FAKE
# * 1 = REAL

# In[12]:


target_variable = news_data['target'].values


# #### Creating dictionaries that map the tokens to the index and the index to the token.

# In[13]:


all_words = []
for l in tokenized_df:
  all_words.extend(l)
all_indices = []
for i in indexed_tokens:
  all_indices.extend(i)

word_to_ix = dict(zip(all_words, all_indices))
ix_to_word = dict(zip(all_indices, all_words))


# #### The BERT algorithm relies on masking to help it learn and to prevent overfitting, so we'll add this to the model.

# In[14]:


mask_variable = [[float(i>0) for i in ii] for ii in index_padded]


# #### This loads the data into train and test dataloaders, which for PyTorch is necessary to iterate through the algorithm.

# In[15]:


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


# #### This is a sample batch from the trainloader. The first tensor contains the embeddings for the articles, the second tensor contains the masking information, and the third tensor contains the target variables for each article.

# In[16]:


next(iter(trainloader))


# 
# ### Now it's time to create the BERT Model!

# #### The BERT model architecture is shown below. This is a BERT base-cased model, which means it has 12 BERT transformer layers, 768 hidden layers, 12 heads, 110M parameters, and is pre-trained on cased English text.
# 

# In[21]:


model = BertForSequenceClassification.from_pretrained('bert-base-cased')
model


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
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {i+1:03d}/{len(trainloader):03d} | '
                  f'Average Loss in last {iteration} iteration(s): {(running_loss/iteration):.4f}')
            running_loss = 0.0
            iteration = 0
        torch.cuda.empty_cache() #memory
        gc.collect() #memory
        losses.append(float(loss.item()))
    with torch.set_grad_enabled(False):
        print(f'\nTraining Accuracy: '
              f'{compute_accuracy(model, trainloader, device):.2f}%')
        


# #### Finally, we score the final model on the test set

# In[ ]:


with torch.set_grad_enabled(False):
  print(f'\n\nTest Accuracy:'
  f'{compute_accuracy(model, testloader, device):.2f}%')


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


X_train_words, X_test_words, y_train_words, y_test_words = train_test_split(news_data['text'], target_variable, 
                                                    test_size=0.1, random_state=42)


# In[ ]:


final_results = X_test_words.to_frame().reset_index(drop=True)
final_results['predicted'] = np.array(test_predictions.reshape(-1), dtype=int).tolist()
final_results['percent'] = np.array(test_predictions_percent.reshape(-1), dtype=float).tolist()
final_results['actual'] = y_test_words
wrong_results = final_results.loc[final_results['predicted']!=final_results['actual']].copy()


# In[ ]:


print('Number of incorrectly classified articles:', len(wrong_results))


# #### This displays the incorrectly predicted instances, along with the percent confidence the algorithm had in each instance. The threshold for classification is 50%. Instances closer to 100% are more confident it's real news and instances closer to 0% are more confident it's fake news.

# In[ ]:


wrong_results.loc[:,'text_short'] = wrong_results.loc[:,'text'].apply(lambda x: x[:500])
wrong_results.loc[:,('text_short', 'percent','predicted','actual')].style.set_properties(subset=['text_short'], **{'width': '1000px', 'white-space':'pre-wrap'})

