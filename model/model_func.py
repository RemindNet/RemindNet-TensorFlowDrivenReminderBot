import tensorflow as tf
from transformers import TFBertModel

def create_bert_model(bert_model_name,task = True,date = True,time = True):
    bert_model = TFBertModel.from_pretrained(bert_model_name)

    # Define input layers
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")

    # Get BERT outputs
    bert_outputs = bert_model(input_ids)[0]  # Output from BERT's last layer

    # Define classification heads for each entity type
    num_labels_task = 3  # "O", "B-task", "I-task"
    num_labels_date = 3  # "O", "B-date", "I-date"
    num_labels_time = 3  # "O", "B-time", "I-time"
    dropout_rate = 0.1

    outputs = []
    # Task classification head
    if task:
        task_logits = tf.keras.layers.Dense(num_labels_task, activation=None, name="task_logits")(bert_outputs)
        task_output = tf.keras.layers.Activation("softmax", name="task_output")(task_logits)
        outputs.append(task_output)

    # # Date classification head
    if date:
        date_logits = tf.keras.layers.Dense(num_labels_date, activation=None, name="date_logits")(bert_outputs)
        date_output = tf.keras.layers.Activation("softmax", name="date_output")(date_logits)
        outputs.append(date_output)

    # # Time classification head
    if time:
        time_logits = tf.keras.layers.Dense(num_labels_time, activation=None, name="time_logits")(bert_outputs)
        time_output = tf.keras.layers.Activation("softmax", name="time_output")(time_logits)
        outputs.append(time_output)

    # Define the model
    model = tf.keras.Model(inputs=input_ids, outputs=outputs)

    return model