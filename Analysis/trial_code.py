import pandas as pd
import re
import requests
import json


def process_reddit(comment):
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


def get_pushshift_data(query, after=None, before=None, sub=None):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000'+'&subreddit=food'
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data

# data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
# data_df = pd.read_csv(data_path + 'reddit_comments_religion_muslims.csv')
#
# data_df['comments_processed'] = data_df['comments'].apply(lambda x: process_reddit(x))
#
# data_df.to_csv(data_path + 'reddit_comments_religion_muslims_processed.csv')


dat = get_pushshift_data('africans')
print(dat)
