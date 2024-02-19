from data_processing.vars import tokenizer
import tensorflow as tf

def pad_labels(labels, max_length):
    return labels + ['O'] * (max_length - len(labels))

def tokenize_input(input_statements):
    return tokenizer(input_statements, add_special_tokens=True, padding='longest', truncation=True,return_tensors="tf")

def encode_labels(labels):
    return 

def convert_tensor(data, label_to_index):
    task_indices = data['task_indices'].tolist()
    time_indices = data['time_indices'].tolist()
    date_indices = data['date_indices'].tolist()
    input_statements = data['input'].tolist()

    encoded_inputs = tokenize_input(input_statements)
    max_length = encoded_inputs['input_ids'][0].shape[0]

    encoded_labels_task = encode_labels(task_indices, max_length, label_to_index)
    encoded_labels_date = encode_labels(date_indices, max_length, label_to_index)
    encoded_labels_time = encode_labels(time_indices, max_length, label_to_index)

    return encoded_inputs, encoded_labels_task, encoded_labels_date, encoded_labels_time

    # task_indices = [eval(elem) for elem in task_indices]
    # time_indices = [eval(elem) for elem in time_indices]
    # date_indices = [eval(elem) for elem in date_indices]    

    # # Pad labels for each entity type
    # max_length = encoded_inputs['input_ids'][0].shape[0]
    # padded_task_indices = [pad_labels(task, max_length) for task in task_indices]
    # padded_date_indices = [pad_labels(date, max_length) for date in date_indices]
    # padded_time_indices = [pad_labels(time, max_length) for time in time_indices]
    
    # # Encode padded labels separately for each entity type
    # encoded_labels_task = [[label_to_index[label] for label in task] for task in padded_task_indices]
    # encoded_labels_date = [[label_to_index[label] for label in date] for date in padded_date_indices]
    # encoded_labels_time = [[label_to_index[label] for label in time] for time in padded_time_indices]

    # # Convert encoded labels to TensorFlow tensors
    # encoded_labels_task = tf.constant(encoded_labels_task)
    # encoded_labels_date = tf.constant(encoded_labels_date)
    # encoded_labels_time = tf.constant(encoded_labels_time)
    # print(check_func == encoded_labels_task)




def encode_labels(labels,max_length,label_to_index):
    labels = [eval(elem) for elem in labels]
    padded_labels = [pad_labels(label, max_length) for label in labels]
    encoded_labels = [[label_to_index[label] for label in label_lst] for label_lst in padded_labels]
    encoded_labels = tf.constant(encoded_labels)
    return encoded_labels
