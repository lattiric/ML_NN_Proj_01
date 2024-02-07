import transformers as hug
import tensorflow as tf
import pandas as pd
import warnings

from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertModel

warnings.filterwarnings("ignore")


TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = TFBertModel.from_pretrained("bert-base-uncased")
def convert_df_to_bert_embedding(df,text_col='text',model=MODEL,tokenizer=TOKENIZER,batch_size=500):
    batches = [(i,min(i+batch_size,len(df))) for i in range(0,len(df),batch_size)] #Split into smaller chunks
    _df = pd.DataFrame()
    max_twt_len = np.max([len(v) for v in df[text_col]])
    print(f'Grabbing BERT Embeddings with padding to {max_twt_len} characters')
    for lower,upper in tqdm(batches):
        chunk = df.iloc[lower:upper]
        features = tokenizer(chunk[text_col].values.tolist(),padding='max_length', truncation=True, return_tensors='tf',max_length=max_twt_len)
        features = model(**features).last_hidden_state[:,0,:]
        chunk['features'] = features.numpy().tolist()
        _df = pd.concat([_df,chunk])
    return _df



if __name__ == "__main__":
    print("In Main")



