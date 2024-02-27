import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
import pickle


# loading model checkpoints

input_path = '/model_checkpoints/mnb_model.pkl'

with open(input_path, 'rb') as file:
    mnb = pickle.load(file)
    

input_path = '/model_checkpoints/ridge_model.pkl'

with open(input_path, 'rb') as file:
    ridge = pickle.load(file)
    
    
input_path = '/model_checkpoints/xgb_model.pkl'

with open(input_path, 'rb') as file:
    xgb = pickle.load(file)

# test set processing
from data_processing import tokenize, vectorize

test = pd.read_csv('/data/test_essays.csv')
bpe_test = tokenize(test)
X_test = vectorize(bpe_test)

# make predictions using classical ML models

mnb_preds = mnb.predict_proba(X_test)[:, 1]
ridge_preds = ridge.predict_proba(X_test)[:, 1]
xgb_preds = xgb.predict_proba(X_test)[:, 1]

classical_ML_preds = 0.3*mnb_preds + 0.4*ridge_preds + 0.4*xgb_preds 

# making predictions using distilroberta

import torch
model_checkpoint = "/model_checkpoints/distilroberta/"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length = 512 , padding=True, truncation=True)
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device);

trainer = Trainer(
    model,
    tokenizer=tokenizer,
)


test_ds = Dataset.from_pandas(test)
test_ds_enc = test_ds.map(preprocess_function, batched=True)
test_preds = trainer.predict(test_ds_enc)
test_logits = test_preds.predictions
distilroberta_preds = sigmoid(test_logits[:,1])


if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_preds = 0.4*classical_ML_preds + 0.6*distilroberta_preds