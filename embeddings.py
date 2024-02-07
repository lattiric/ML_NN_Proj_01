import transformers as hug
import tensorflow as tf
import pandas as pd
import warnings
import helpers
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertModel
from multiprocessing import Pool
from multiprocessing import cpu_count
import string

embeddingz = None #This is horrible practice but idgaf

def _convert_sentence_to_embedding(sentence):
        features = []
        for word in sentence.split(' '):
            word = word.translate(str.maketrans('', '', string.punctuation)).lower() # Remove punct
            try:
                features = features + list(globals()['embeddingz'].loc[word])
            except KeyError as e:
                pass
        if len(features) == 0: #If we couldnt convert, set to None to remove later
            features = None
        return features

warnings.filterwarnings("ignore")
class EmbeddingGenerator:

    def __init__(self,bert_model=None,bert_tokenizer=None,embeddings=None,embedding_path='../../data/concept_net/glove.840B.300d.txt'):
        if bert_model: 
            self.model = bert_model
        else: 
            self.model = TFBertModel.from_pretrained("bert-base-uncased")

        if bert_tokenizer: 
            self.tokenizer = bert_tokenizer
        else: 
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if embeddings:
            self.embeddings = embeddings
            EMBEDDINGS = embeddings.copy()
        else:
            print(f"{'=' * 150}\nLoading embeddings... This may take a while")
            self.embeddings = helpers.load_embeddings(embedding_path)
            

    def convert_df_to_bert_embedding(self,df,text_col='text',batch_size=500):
        batches = [(i,min(i+batch_size,len(df))) for i in range(0,len(df),batch_size)] #Split into smaller chunks
        _df = pd.DataFrame()
        max_twt_len = np.max([len(v) for v in df[text_col]])
        print(f'Grabbing BERT Embeddings with padding to {max_twt_len} characters')
        for lower,upper in tqdm(batches):
            chunk = df.iloc[lower:upper]
            features = self.tokenizer(chunk[text_col].values.tolist(),padding='max_length', truncation=True, return_tensors='tf',max_length=max_twt_len)
            features = self.model(**features).last_hidden_state[:,0,:]
            chunk['features'] = features.numpy().tolist()
            _df = pd.concat([_df,chunk])
        return _df

    def convert_df_single_word_to_other_embedding(self,df,text_col='text',verbose=False):
        _df = df.copy()
        globals()['embeddingz'] = self.embeddings.copy()
        features = []
        for word in _df[text_col]:
            try:
                features.append(self.embeddings.loc[word])
            except KeyError as e:
                features.append(None)
                if verbose:
                    print(f'\"{word}\" not in embedding')
        _df['features'] = features
        print(f'Removing {len(_df[_df["features"].isnull()])} words that did not exist in embeddings') #TODO Fix array vs no arrays

        return _df.drop(_df.loc[_df['features'].isnull()].index, inplace=False)
    
    def convert_df_multi_word_to_other_embedding(self,df,text_col='text'):
        print('Converting multi word text into Other Embedding (Glove by default)')
        _df = df.copy()
        with Pool(cpu_count()) as p:
            features = list(tqdm(p.imap(_convert_sentence_to_embedding,_df[text_col]),total=len(_df)))            
        _df['features'] = features
        return _df

    

if __name__ == "__main__":
    class tmp:
        def __init__(self,attr):
            self.attr = attr
        def outer(self,vals):
            with Pool(5) as p:
                a = p.map(self.inner,vals)
                print(a)
        def inner(self,sentence):
            attrs = []
            for word in sentence.split(' '):
                print(word)
                attrs = attrs + self.attr[word]
            return attrs

    c = tmp(attr={'hello':[1,2,3],'my':[4,5,6],'name':[7,8,9]})
    c.outer(vals=['hello hello my my','name name my my'])


