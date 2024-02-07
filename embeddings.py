
import pandas as pd
import warnings
import helpers
from tqdm.auto import tqdm
import numpy as np

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from multiprocessing import Pool
from multiprocessing import cpu_count
import string
from functools import lru_cache

embeddingz = None #This is horrible practice but idgaf

@lru_cache
def embedding_for_word(word):
    return list(globals()['embeddingz'].loc[word])


def _convert_sentence_to_embedding(sentence):
        features = []
        for word in sentence.split(' '):
            word = word.translate(str.maketrans('', '', string.punctuation)).lower() # Remove punct
            try:
                features = features + embedding_for_word(word)
            except KeyError as e:
                pass
        if len(features) == 0: #If we couldnt convert, set to None to remove later
            features = None
        return features

warnings.filterwarnings("ignore")
class EmbeddingGenerator:

    def __init__(self,bert_model=None,bert_tokenizer=None,embeddings=None,embedding_path='../../data/concept_net/glove.840B.300d.txt',only_bert=False,cached_embedding_path=None):
        if bert_model: 
            self.model = bert_model
        else: 
            self.model = TFBertModel.from_pretrained("bert-base-uncased")

        if bert_tokenizer: 
            self.tokenizer = bert_tokenizer
        else: 
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if not only_bert:
            if cached_embedding_path:
                self.embeddings = pd.read_parquet(cached_embedding_path)
            else:
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
        # globals()['embeddingz'] = self.embeddings.copy()
        features = []
        for word in tqdm(_df[text_col]):
            try:
                features.append(embedding_for_word(word))
            except KeyError as e:
                features.append(None)
                if verbose:
                    print(f'\"{word}\" not in embedding')
        _df['features'] = features
        print(f'Removing {len(_df[_df["features"].isnull()])} words that did not exist in embeddings') #TODO Fix array vs no arrays

        return _df.drop(_df.loc[_df['features'].isnull()].index, inplace=False)
    
    def convert_df_multi_word_to_other_embedding(self,df,text_col='text'):
        print('Converting multi word text into Other Embedding (Glove by default)')
        globals()['embeddingz'] = self.embeddings.copy()
        _df = df.copy()
        with Pool(cpu_count()) as p:
            features = list(tqdm(p.imap(_convert_sentence_to_embedding,_df[text_col]),total=len(_df)))            
        _df['features'] = features

        print(f'Removing {len(_df[_df["features"].isnull()])} entries that did not exist in embeddings') #TODO Fix array vs no arrays

        return _df.drop(_df.loc[_df['features'].isnull()].index, inplace=False)

    def get_train_test_val_with_bert(self,df,text_col='text',batch_size=500,test_size=None,train_size=None,random_state=None,shuffle=True,stratify=None,n_components=300):
        """ Returns train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text
        Will pca decompose all feature vectors based ONLY on train features
        """
        _df = df.copy()
        _df = self.convert_df_to_bert_embedding(_df,text_col=text_col,batch_size=batch_size)
        x = np.array([x for x in _df['features']])
        train_features,test_features,train_labels,test_labels,train_text,test_text = train_test_split(x,_df['target'],_df['text'],test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)
        train_features,val_features,train_labels,val_labels,train_text,val_text = train_test_split(train_features,train_labels,train_text,test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)

        pca = PCA(n_components=n_components)
        train_features = pca.fit_transform(train_features)
        val_features = pca.transform(val_features)
        test_features = pca.transform(test_features)



        return train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text
    
    def get_train_test_val_with_other_embedding_multi_word(self,df,text_col='text',batch_size=500,test_size=None,train_size=None,random_state=None,shuffle=True,stratify=None,n_components=300):
        """ Returns train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text
        Will pca decompose all feature vectors based ONLY on train features
        """
        _df = df.copy()
        _df = self.convert_df_multi_word_to_other_embedding(_df,text_col=text_col)

        max_vec_len = np.max([len(v) for v in _df['features']])

        print(f'Max feature vec size: {max_vec_len}')
        _df['features'] = [v + [0] * (max_vec_len - len(v)) for v in _df['features']]

        x = np.array([x for x in _df['features']])

        train_features,test_features,train_labels,test_labels,train_text,test_text = train_test_split(x,_df['target'].to_numpy(),_df['text'].to_numpy(),test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)
        train_features,val_features,train_labels,val_labels,train_text,val_text = train_test_split(train_features,train_labels,train_text,test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)

        pca = PCA(n_components=n_components)
        train_features = pca.fit_transform(train_features)
        val_features = pca.transform(val_features)
        test_features = pca.transform(test_features)
        return train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text



    
    def get_train_test_val_with_other_embedding_single_word(self,df,text_col='text',batch_size=500,test_size=None,train_size=None,random_state=None,shuffle=True,stratify=None,n_components=300):
        """ Returns train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text
        """
        _df = df.copy()
        _df = self.convert_df_single_word_to_other_embedding(_df,text_col=text_col)

        x = np.array([x for x in _df['features']])
        train_features,test_features,train_labels,test_labels,train_text,test_text = train_test_split(x,_df['target'],_df['text'],test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)
        train_features,val_features,train_labels,val_labels,train_text,val_text = train_test_split(train_features,train_labels,train_text,test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)

        return train_features,test_features,val_features,train_labels,test_labels,val_labels,train_text,test_text,val_text


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


