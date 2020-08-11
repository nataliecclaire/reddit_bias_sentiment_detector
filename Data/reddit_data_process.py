import pandas as pd
import re


def process_tweet(sent):
    # special cases - 15959
    # print(sent)
    sent = sent.encode("ascii", errors="ignore").decode() # check this output
    # print(sent)
    sent = re.sub('@[^\s]+', '', sent)
    sent = re.sub('https: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http[^\s]+', '', sent)

    # split camel case combined words
    sent = re.sub('([A-Z][a-z]+)', r'\1', re.sub('([A-Z]+)', r' \1', sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(' \d+', '', sent)
    # remove words with letter+number
    sent = re.sub('\w+\d+|\d+\w+', '', sent)

    # remove spaces
    sent = re.sub('[\s]+', ' ', sent)
    sent = re.sub('[^\w\s.!\-?]', '', sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(" rt ", "", sent)

    sent = re.sub('- ', '', sent)

    sent = sent.strip()

    # print(sent)
    return sent


def process_reddit(comment):
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race'  # 'race' #'gender' # 'religion'
demo_1 = 'black'  # 'jews' # 'black' #'female' # 'jews'
demo_2 = 'white'

colNames = ('id', 'comments', 'comments_processed')

demo1_df_processed = pd.DataFrame(columns=colNames)
df_list = []
for i in range(5):
    demo1_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_raw_' + str(i)+'.csv')
    demo1_df = demo1_df.loc[:, ~demo1_df.columns.str.contains('^Unnamed')]

    demo1_df['comments_processed'] = demo1_df['comments'].apply(lambda x: process_tweet(x))
    # print('Before length filter {}'.format(demo1_df.shape))
    demo1_df = demo1_df[demo1_df['comments_processed'].str.len() < 150]
    # pd.concat([demo1_df_processed, demo1_df])
    # print('After length filter {}'.format(demo1_df.shape))
    # demo1_df_processed.append(demo1_df, ignore_index=True)
    df_list.append(demo1_df)

demo1_df_processed = pd.concat(df_list, ignore_index=True)
print(demo1_df_processed.shape)
demo1_df_processed.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv', index=False)

'''
ethnicity_pairs = {}
with open(data_path + 'religion_opposites_jc.txt') as f:
    for line in f:
        (key, val) = line.split(',')
        key = key.replace('"', '').replace('\n', '')
        val = val.replace('"', '').replace('\n', '')
        ethnicity_pairs[key] = val

print(len(ethnicity_pairs))
'''
demo2_df = pd.DataFrame(columns=['initial_demo', 'replaced_demo', 'comments', 'comments_processed'])
pairs = (('black', 'white'), ('african american', 'anglo american'), ('african-american', 'anglo-american'),
         ('afro-american', 'anglo-american'), ('african', 'american'), ('afroamericans', 'angloamericans'), ('negroes', 'caucasians'), ('dark-skin', 'light-skin'),
         ('dark skin', 'light skin'))

for idx, row in demo1_df_processed.iterrows():
    initial_demo = []
    replaced_demo = []
    s = row['comments_processed']
    demo2_df.at[idx, 'comments'] = s

    for p in pairs:
        # s = s.replace(*p)
        if p[0] == 'african' and p[0] in s and ('anglo american' in s or 'anglo-american' in s):
            s = s.replace(*p)
        elif p[1] == 'american' and p[1] in s and ('anglo american' in s or 'anglo-american' in s):
            s = s.replace(*p)
        elif p[0] == 'afro-american' and p[0] in s:
            s = s.replace(*p)
        else:
            s = s.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])
        if p[1] in s:
            initial_demo.append(p[0])
            replaced_demo.append(p[1])
    demo2_df.at[idx, 'comments_processed'] = s
    demo2_df.at[idx, 'initial_demo'] = initial_demo
    demo2_df.at[idx, 'replaced_demo'] = replaced_demo


print(demo2_df.shape)
demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed' + '.csv', index=False)