from data_processing.tensor_convert import encoded_inputs, encoded_labels_task, encoded_labels_date, encoded_labels_time
from data_processing.vars import bert_model_name
from model.model_func import create_bert_model
import tensorflow as tf

model = create_bert_model(bert_model_name,task = True,date =True,time = True)

model.compile(optimizer='adam',
              loss={'task_output': 'sparse_categorical_crossentropy',
                    'date_output': 'sparse_categorical_crossentropy',
                    'time_output': 'sparse_categorical_crossentropy'},
              metrics={'task_output': 'accuracy',
                       'date_output': 'accuracy',
                       'time_output': 'accuracy'})

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

print(model.summary())

print(encoded_inputs['input_ids'])
history = model.fit(
    encoded_inputs['input_ids'],
    {'task_output': encoded_labels_task, 'date_output': encoded_labels_date, 'time_output': encoded_labels_time},
    epochs=50,
)