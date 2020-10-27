'''
# get test data for both target groups: used for measuring bias
df_demo2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed_phrase_biased' + '.csv')
df_demo2 = df_demo2.rename(columns={'comments': 'comments_merge_on'})

df_demo2 = df_demo2.drop(columns=['initial_demo', 'replaced_demo'])

df_test = df_test.drop_duplicates(subset='comments_processed')
df_test = df_test.rename(columns={'comments_processed': 'comments_merge_on'})
print(df_test.head)
print(df_demo2.head())

df_test_demo2 = df_test.merge(df_demo2, on='comments_merge_on', how='inner')

df_test_demo2 = df_test_demo2.drop_duplicates(subset='comments_merge_on')

print(df_test.shape)
print(df_test_demo2.shape)
df_test = df_test.rename(columns={'comments_merge_on': 'comments_processed'})
df_test_demo2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed_phrase_biased_testset' + '.csv', index=False)
'''

'''
demo_1 = 'jews' # 'white'
demo_2 = 'christians'

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
'''