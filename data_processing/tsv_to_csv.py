import csv

tsv_file = 'fb_data/train-en.tsv'
csv_file = 'fb_data/train_en.csv'

def tsv_to_csv(tsv_file, csv_file):
    with open(tsv_file, 'r') as tsvfile, open(csv_file, 'w', newline='') as csvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        csv_writer = csv.writer(csvfile, delimiter=',')

        for row in tsv_reader:
            csv_writer.writerow(row)

tsv_to_csv(tsv_file, csv_file)