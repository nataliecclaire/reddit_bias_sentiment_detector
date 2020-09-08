import pandas as pd
from sklearn.model_selection import train_test_split
import re


def build_dataset(df, demo, dest_path):
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        bos_token = '<bos>'
        eos_token = '<eos>'
        comment = row['comments_1']
        if 'pos' in demo:
            bias_token = 'unbiasannot'
            if str(comment) != 'nan':
                data += bias_token + ' ' + bos_token + ' ' + comment + ' ' + eos_token + '\n'
            else:
                data += bias_token + ' ' + bos_token + ' ' + str(comment) + ' ' + eos_token + '\n'
        else:
            bias_token = 'biasannot'
            # if row['diff_perplex'] > 0:
            #     data += str(0) + ' ' + bos_token + ' ' + comment + ' ' + eos_token + '\n'
            # else:
            data += bias_token + ' ' + bos_token + ' ' + comment + ' ' + eos_token + '\n'

    f.write(data)


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender' #  # 'race'  # 'race' #'gender' # 'religion'

demo_1 = 'black'
df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff' + '.csv')
train_test_ratio = 0.9
df_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

desti_path = data_path + 'bias_annotated/'
build_dataset(df_train, demo_1, desti_path + demo_1 + 'bias_annotated_train_2.txt')
build_dataset(df_test, demo_1, desti_path + demo_1 + 'bias_annotated_valid_2.txt')

demo_1 = 'black_pos'
df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + demo_1 + '_diff' + '.csv')
train_test_ratio = 0.9
df_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

desti_path = data_path + 'bias_annotated/'
build_dataset(df_train, demo_1, desti_path + demo_1 + 'bias_annotated_train_2.txt')
build_dataset(df_test, demo_1, desti_path + demo_1 + 'bias_annotated_valid_2.txt')