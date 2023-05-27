import csv
import os

def read_csv_to_list(csv_file, num_rows):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            float_row = [float(element) for element in row]
            data.append(float_row)
            if i + 1 >= num_rows:
                break
    return data

def replicate_string(text, length):
    return [text] * length

def prepare_dataset(train_size=300, test_size=30, data_directory="your/directory"):
    X,y = [],[]
    for file in os.listdir(data_directory):
        X+=read_csv_to_list(os.path.join(data_directory, file), train_size+test_size)
        y+=replicate_string((str(file)[:-4]), train_size+test_size)

    return X,y