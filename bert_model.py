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
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BertModel():
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('blahtoken')
		self.model = BertForSequenceClassification.from_pretrained('blah')

	def predict(self, article):
		test_news_data = pd.DataFrame({"text": article}, index=[0])
		tokenized_df = list(map(lambda t: ['[CLS]'] + self.tokenizer.tokenize(t)[:510] + ['[SEP]'], test_news_data['text']))
		totalpadlength = 512
		indexed_tokens = list(map(self.tokenizer.convert_tokens_to_ids, tokenized_df))
		index_padded = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in indexed_tokens])
		all_words = []
		for l in tokenized_df:
		  all_words.extend(l)
		all_indices = []
		for i in indexed_tokens:
		  all_indices.extend(i)

		word_to_ix = dict(zip(all_words, all_indices))
		ix_to_word = dict(zip(all_indices, all_words))
		mask_variable = [[float(i>0) for i in ii] for ii in index_padded]
		index_padded = torch.from_numpy(index_padded)
		index_padded = index_padded.long()
		mask_variable = torch.tensor(mask_variable)
		_, yhat = self.model(input_ids=index_padded, attention_mask=mask_variable, labels=labels)
		prediction = (torch.sigmoid(yhat[:,1]) > 0.5).long().view(-1,1)
		return prediction.item()