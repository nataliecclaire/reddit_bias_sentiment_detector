import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset(df, demo, dest_path):
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        bos_token = '<bos>'
        eos_token = '<eos>'
        comment = row['comments_2']

        data += bos_token + ' ' + comment + ' ' + eos_token + '\n'

    f.write(data)


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'race' # 'gender' #  # 'race'  # 'race' #'gender' # 'religion'

demo_1 = 'white'
df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff' + '.csv')
train_test_ratio = 0.9
df_train, df_test = train_test_split(df, train_size=train_test_ratio, random_state=1)

print('Train {}'.format(df_train.shape))
print('Test {}'.format(df_test.shape))

desti_path = data_path + 'bias_annotated/'
build_dataset(df_train, demo_1, desti_path + demo_1 + '_target2_train.txt')
build_dataset(df_test, demo_1, desti_path + demo_1 + '_target2_valid.txt')

print('Saving test dataset...')
df_test.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff_test' + '.csv', index=False)