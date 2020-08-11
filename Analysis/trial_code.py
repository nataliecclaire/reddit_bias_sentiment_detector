import pandas as pd
import re


def process_reddit(comment):
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
data_df = pd.read_csv(data_path + 'reddit_comments_religion_muslims.csv')

data_df['comments_processed'] = data_df['comments'].apply(lambda x: process_reddit(x))

data_df.to_csv(data_path + 'reddit_comments_religion_muslims_processed.csv')