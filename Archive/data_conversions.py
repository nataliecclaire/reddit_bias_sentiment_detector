import csv

csv_file = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dialog_response_DialoGPT-small.csv'
tsv_file = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dialog_response_DialoGPT-small.tsv'
tsv_file_mask = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dialog_response_DialoGPT-small.tsv.XYZ'

csv_reddit = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/RedditComments.csv'
tsv_reddit = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/RedditComments.tsv'
tsv_mask_reddit = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/RedditComments.tsv.XYZ'


def csv_to_tsv(csv_f, tsv_f):

    with open(csv_f, 'r', encoding='utf-8') as csvin, open(tsv_f, 'w', encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout)

        for row in csvin:
            # print(row)
            for colm in row[1:7]:
                # print(colm)
                tsvout.writerow([colm])


def csv_to_tsv_mask(csv_f, tsv_f):

    with open(csv_f, 'r', encoding='utf-8') as csvin, open(tsv_f, 'w', encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout)

        for row in csvin:
            # print(row)
            for colm in row[1:7]:
                # print(colm)
                for demo in ['men', 'women', 'black people', 'white people', 'gay people', 'straight people']:
                    if demo in colm:
                        colm = colm.replace(demo, 'XYZ')
                        tsvout.writerow([colm])


def csv_to_tsv_reddit(csv_f, tsv_f):

    with open(csv_f, 'r', encoding='utf-8') as csvin, open(tsv_f, 'w', encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout)

        for row in csvin:
            print(row[3])
            tsvout.writerow([row[3]])


def csv_to_tsv_mask_reddit(csv_f, tsv_f):

    with open(csv_f, 'r', encoding='utf-8') as csvin, open(tsv_f, 'w', encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout)

        for row in csvin:
            # print(row)
            for demo in ['men', 'women', 'Black people', 'white people', 'gay people', 'straight people']:
                if demo in row[3]:
                    colm = row[3].replace(demo, 'XYZ')
                    tsvout.writerow([colm])


# csv_to_tsv(csv_file, tsv_file)
# csv_to_tsv_mask(csv_file, tsv_file_mask)
# csv_to_tsv_reddit(csv_reddit, tsv_reddit)
csv_to_tsv_mask_reddit(csv_reddit, tsv_mask_reddit)