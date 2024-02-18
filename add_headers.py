import csv

def add_headers(file_path, headers):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

# Usage example
file_path = '/path/to/your/csv/file.csv'
headers = ['Header1', 'Header2', 'Header3']
add_headers(file_path, headers)
