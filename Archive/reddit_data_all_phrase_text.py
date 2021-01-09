import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset_manual_annot(df, dest_path, column):
    f = open(dest_path, 'w')
    data = ''

    for idx, row in df.iterrows():
        comment = row[column]
        if column == 'bias_phrase':
            data += str(comment) + '\n'
        else:
            data += comment + '\n'

    f.write(data)


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'religion1' # 'race' # 'gender' #  # 'race'  # 'race' #'gender' # 'religion'

demo_1 = 'jews'
demo_2 = 'christians'

df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed_phrase_annotated' + '.csv', encoding='Latin-1')
df = df.dropna(subset=['bias_phrase', 'phrase'])

train_test_ratio = 0.7
df_train, df_test = train_test_split(df, train_size=train_test_ratio, random_state=1)

print('Train {}'.format(df_train.shape))
print('Test {}'.format(df_test.shape))

desti_path = data_path + 'bias_annotated/'
df_test = df_test[df_test['bias_phrase'] == 1]
build_dataset_manual_annot(df_train, desti_path + demo + '_bias_unbias_manual_train.txt', column='phrase')
build_dataset_manual_annot(df_test, desti_path + demo + '_bias_unbias_manual_test.txt', column='phrase')

build_dataset_manual_annot(df_train, desti_path + demo + '_bias_unbias_manual_train_label.txt', column='bias_phrase')
