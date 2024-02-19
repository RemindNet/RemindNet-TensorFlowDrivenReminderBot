import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = pd.read_csv('set_index.csv')
label_to_index = {"O": 0, "B-task_detail": 1, "I-task_detail": 2, "B-date": 3, "I-date": 4, "B-time": 5, "I-time": 6}
bert_model_name = "bert-base-uncased"
