import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

train = pd.read_csv('/data/train_essays.csv')

train = train.drop_duplicates(subset=['text'])
train = train.reset_index(drop=True)

def train_and_tokenize(train, LOWERCASE=False, VOCAB_SIZE=30522):
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    dataset = Dataset.from_pandas(train[['text']])

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


    tokenized_texts_train = [tokenizer.tokenize(text) for text in train['text'].tolist()]
    

    return tokenized_texts_train

bpe_tokenized_texts_train = train_and_tokenize(train)

def dummy(text):
    return text

def vectorize_texts(tokenized_texts):
    
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, 
                                 analyzer='word', tokenizer=dummy, preprocessor=dummy, 
                                 token_pattern=None, strip_accents='unicode', min_df=2, max_features=5000000)


    tf_train = vectorizer.fit_transform(tokenized_texts)


    return tf_train

tf_train = vectorize_texts(bpe_tokenized_texts_train)

from scipy.sparse import save_npz, load_npz

save_npz('/data/processed_train.npz', tf_train)

tf_train = load_npz('/data/processed_train.npz')