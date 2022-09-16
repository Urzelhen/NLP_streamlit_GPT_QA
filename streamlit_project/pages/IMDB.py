import streamlit as st
from string import punctuation, digits
import joblib

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

# natural language toolkit 
import nltk

# regular expression
import re

import string

import tqdm
from tqdm.notebook import tqdm

from collections import Counter

import torch
from torch import nn
from torch import optim

from torchsummary import summary
from torchmetrics import MeanSquaredError
from nltk.corpus import stopwords


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


from catboost import CatBoostClassifier

st.sidebar.markdown("IMDB")

st.image("/home/valentina/ds_offline/learning/12-nlp/streamlit_project_others/IMDB/imdb.jpg",
        width=400,
        )


review_text = st.text_input('Напишите отзыв')

st.write("""
### ML
""")
review_text_1 = pd.Series(review_text)
def clean(text):
        text = text.lower() # нижний регистр
        text = re.sub(r'http\S+', " ", text) # удаляем ссылки
        text = re.sub(r'@\w+',' ',text) # удаляем упоминания пользователей
        text = re.sub(r'#\w+', ' ', text) # удаляем хэштеги
        text = re.sub(r'\d+', ' ', text) # удаляем числа
        text = re.sub(r'<.*?>',' ', text) # 
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

review_text_cleaned = review_text_1.apply(clean)
    
wn_lemmatizer = WordNetLemmatizer()
lemmatized_text = []
for review in review_text_cleaned:
        lemmatized_text.append(' '.join([wn_lemmatizer.lemmatize(word) for word in review.split()]))

reg_tokenizer = RegexpTokenizer('\w+')
tokenized_text = reg_tokenizer.tokenize_sents(lemmatized_text)
sw = stopwords.words('english')
clean_tokenized_reviews = [] 
for element in tokenized_text:
    clean_tokenized_reviews.append(' '.join([word for word in element if word not in sw]))

model = joblib.load('/home/valentina/ds_offline/learning/12-nlp/streamlit_project/pages/cat_class.pkl')
matrixer = joblib.load('/home/valentina/ds_offline/learning/12-nlp/streamlit_project/pages/cvec_representation.pkl')
predict = model.predict_proba(matrixer.transform(clean_tokenized_reviews))
st.write(predict)


st.write("""
### LSTM
""")
device = 'cpu'
class sentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    
    def __init__(self,
                 vocab_size, # объём словаря с которым мы работаем
                 output_size, # нейроны полносвязного
                 embedding_dim, # размер выходного эмбеддинга
                 hidden_dim, # размерность внутреннего слоя LSTM
                 n_layers, # число слоев в LSTM
                 drop_prob=0.5):
        
        super().__init__()
        
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # nn.Linear(64, 16) / embedding_dim - выходная размерность 
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            dropout=drop_prob, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):

        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        # print(f'Embed shape: {embeds.shape}')
        lstm_out, hidden = self.lstm(embeds, hidden)
        # print(f'lstm_out {lstm_out.shape}')
        # print(f'hidden {hidden[0].shape}')
        # print(f'hidden {hidden[1].shape}')
        #stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # print(f'lstm out after contiguous: {lstm_out.shape}')
        # Dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        #sigmoid function
        sig_out = self.sigmoid(out)
        
        # reshape to be batch size first
        # print(sig_out.shape)
        sig_out = sig_out.view(batch_size, -1)
        # print(sig_out.shape)
        # print(f'Sig out before indexing:{sig_out.shape}')
        sig_out = sig_out[:, -1] # get last batch of labels
        # print(sig_out.shape)
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        ''' Hidden state и Cell state инициализируем нулями '''

        h0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

model_loaded = sentimentLSTM(161203, 1, 32, 16, 2)
model_loaded.load_state_dict(torch.load('/home/valentina/ds_offline/learning/12-nlp/streamlit_project/pages/state_dict.pt'))

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import csv

def LSTMpred(str: str):
    df = pd.DataFrame(pd.Series(str, name='review'))
    # print(df)
    def clean(text):
        text = text.lower() # нижний регистр
        # text = re.sub(r'http\S+', " ", text) # удаляем ссылки
        # text = re.sub(r'@\w+',' ',text) # удаляем упоминания пользователей
        # text = re.sub(r'#\w+', ' ', text) # удаляем хэштеги
        text = re.sub(r'\d+', ' ', text) # удаляем числа
        text = re.sub(r'<.*?>',' ', text) # 
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    df['review'] = df['review'].apply(clean)

    wn_lemmatizer = WordNetLemmatizer()

    lemmatized_text = []

    for review in df['review']:
        lemmatized_text.append(' '.join([wn_lemmatizer.lemmatize(word, 'a') for word in review.split()]))

    reg_tokenizer = RegexpTokenizer('\w+')
    
    tokenized_text = reg_tokenizer.tokenize_sents(lemmatized_text)
    sw = stopwords.words('english')
    # print(sw)
    clean_tokenized_reviews = [] 
    for i, element in tqdm(enumerate(tokenized_text), total=len(tokenized_text)):
        clean_tokenized_reviews.append(' '.join([word for word in element if word not in sw]))
    df['review'] = pd.Series(clean_tokenized_reviews)
    # print(df)

    
    corpus = [word for text in df['review'] for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()

    

    with open('/home/valentina/ds_offline/learning/12-nlp/streamlit_project/pages/vocab.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # print(reader)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            vocab_to_int = {rows[0]:rows[1] for rows in reader}
    
    reviews_int = []
    for text in df['review']:
        # print(text)
        r = [int(vocab_to_int[word]) for word in text.split()]
       
        reviews_int.append(r)

    def padding(review_int, seq_len):
        '''
        Делаем padding, если длинна меньше seq_len, 
        если больше – берем первые seq_len индексов
        '''
        features1 = np.zeros((len(reviews_int), seq_len), dtype = int)
        for i, review in enumerate(review_int):
            if len(review) <= seq_len:
                zeros = list(np.zeros(seq_len - len(review)))
                new = zeros + review
            else:
                new = review[: seq_len]
            # print(i, new)
            features1[i, :] = np.array(new)
                
        return features1
    features = padding(reviews_int, seq_len = 50)
    return np.array(features)
    
for_pred = LSTMpred(review_text)

test_h1 = model_loaded.init_hidden(1)
# print(test_h1)

model_loaded.eval()
# for inputs, labels in test_loader:
    #     print(inputs)
test_h = tuple([each.data for each in test_h1])

# inputs, labels = inputs.to(device), labels.to(device)

output, test_h = model_loaded(torch.tensor(for_pred), test_h)

# test_loss = criterion(output.squeeze(), labels.float())
# test_losses.append(test_loss.item())
# sm = torch.nn.Softmax()

pred = float(output.squeeze().detach().numpy())
# print(pred, pred.shape)  

output = {'Positive': f'{format(pred*100, ".2f")} %', 'Negative': f'{format((1-pred)*100, ".2f")} %' }
out_stl = pd.DataFrame(output, index=['Probability'])

st.write(out_stl)

st.write("""
### BERT + ML
""")

