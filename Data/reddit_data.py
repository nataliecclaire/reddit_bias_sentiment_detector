import json
import requests
import pandas as pd
import time
import re
import multiprocessing as mp
from collections import defaultdict


def get_pushshift_data(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def get_pushshift_comments(q1, q2, s, b, a):
    url = 'https://api.pushshift.io/reddit/search/comment/?q='+str(q1)+'+'+str(q2)+'&size='+str(s)+'&after='+str(a)+'&before='+str(b)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


# sub='PS4'
# before = "1538352000" #October 1st
# after = "1514764800"  #January 1st
# query = "Screenshot"
# red_data = getPushshiftData(query, after, before, sub)
# print(red_data)


def get_reddit_comments(qd):
    global comments_dict
    for qf in query_feature:
        before = int(time.time())
        for c in range(chunks):
            try:
                after = before - (60 * 60 * 24 * 30)
                red_comments = get_pushshift_comments(qd, qf, size, before, after)
                # print(red_comments)
                # print(len(red_comments))
                time.sleep(5)
                for idx, com in enumerate(red_comments):
                    # if len(com['body']) <= length:
                    comments_dict[com['id']] = com['body']
                        # print(com['body'])
                        # print(len(com['body']))
                        # comments_df.at[i, 'comments'] = com['body']
                        # comments_df.at[i, 'id'] = com['id']
                        # comments_df.at[i, 'created_utc'] = com['created_utc']
                        # i += 1

                        # if "comments" in comments_dict:
                        #     comments_dict["comments"].append(com['body'])
                        # else:
                        #     comments_dict["comments"] = com['body']

                        # comments_dict.setdefault("comments", []).append(com['body'])

                before = after
            except Exception as e:
                print('Exception observed: {}'.format(repr(e)))



if __name__ == '__main__':

    start = time.time()
    # query_demo = ['"Women are"', '"Women should"', '"Mothers are"', '"His daughter is"']
    # query_feature = ['adoption', 'adoptive', 'birth', 'bride', 'care-giver', 'child', 'childhood', 'children', 'clan',
    #                  'cousin', 'devoted', 'divorce', 'engaged', 'engagement', 'estranged', 'faithful', 'family',
    #                  'fiancee',
    #                  'folks', 'foster', 'helpmate', 'heritage', 'household', 'in-law', 'infancy', 'infant', 'inherit',
    #                  'inheritance', 'kin', 'kindred', 'kinfolk', 'kinship', 'kith', 'lineage', 'love', 'marry',
    #                  'marriage',
    #                  'mate', 'maternal', 'matrimony', 'natal', 'newlywed', 'nuptial', 'offspring', 'orphan',
    #                  'parent relative',
    #                  'separation', 'sibling', 'spouse', 'tribe', 'triplets', 'twins', 'wed', 'wedding', 'wedlock']

    data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
    demo = 'religion2' # 'race' #'gender' # 'religion'
    demo_1 = 'muslims' # 'jews' # 'black' #'female' # 'jews'

    with open(data_path + demo + '_' + demo_1 + '.txt') as f:
        query_feature = [line.split('\n')[0] for line in f]

    with open(data_path + demo + '_opposites.txt') as f:
        query_demo = [line.split(',')[0] for line in f]

    print(query_demo)
    print(query_feature)

    size = 400
    chunks = 40
    length = 150

    loops = int(len(query_demo)/4) if len(query_demo) % 4 == 0 else int(len(query_demo)/4) + 1
    print('Looping {} times'.format(loops))

    for i in range(loops):
        manager = mp.Manager()
        comments_dict = manager.dict()

        query_demo_4 = query_demo[i*4:i*4+4]

        with mp.Pool(processes=4) as pool:
            pool.map(get_reddit_comments, query_demo_4)

        # print(comments_dict)
        # comments_dict = dict(comments_dict)
        # print(comments_dict)

        comments_df = pd.DataFrame(list(comments_dict.items()), columns=['id', 'comments'])
        # print(type(red_comments))
        # comments_df['comments_processed'] = comments_df['comments'].apply(lambda sent: process_reddit(sent))
        comments_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_raw_' + str(i) + '.csv')

        print('Total time for code execution: {} of iteration {}'.format((time.time() - start)/60, i))
