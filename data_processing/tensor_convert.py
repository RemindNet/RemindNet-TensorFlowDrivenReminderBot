from data_processing.preprocess_func import convert_tensor
from data_processing.vars import data, label_to_index

# Convert the data to TensorFlow tensors
encoded_inputs, encoded_labels_task, encoded_labels_date, encoded_labels_time = convert_tensor(data, label_to_index)

if len(encoded_inputs['input_ids']) == len(encoded_labels_task) == len(encoded_labels_date) == len(encoded_labels_time):
    print("Lengths of encoded inputs and labels are equal.")
elif len(encoded_inputs['input_ids']) != len(encoded_labels_task) != len(encoded_labels_date) != len(encoded_labels_time):
    print(len(encoded_inputs['input_ids']), len(encoded_labels_task), len(encoded_labels_date), len(encoded_labels_time))
    raise ValueError("Lengths of encoded inputs and labels are not equal.")

print(type(encoded_inputs['input_ids']))