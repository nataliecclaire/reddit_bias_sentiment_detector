import pandas as pd
import numpy as np
from scipy import stats
from utils import helper_functions as helpers
from transformers import AutoModelWithLMHead, AutoTokenizer
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging


def get_perplexity_list(df, m, t):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = helpers.score(row['comments_processed'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = helpers.score(row['comments_1'], m, t)
            else:
                perplexity = helpers.score(row['comments_2'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'

ON_SET = True
GET_PERPLEXITY = False
ON_TESTSET = False
GET_PERPLEXITY_TEST = False

demo = 'religion1' # 'race' # '' # 'religion2' # 'gender' # 'race' #'orientation'  #
demo_1 = 'jews' # 'black' #  # 'muslims' # 'female' # 'black_pos' # 'muslims' #  # 'lgbtq'
demo_2 = 'christians' # 'white'  # 'male' # 'white_pos'  # 'white' #'straight'# #'christians2'

if ON_SET:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'_test.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 600)
pd.options.display.max_columns = 10

if ON_SET:
    if GET_PERPLEXITY:

        logging.info('Calculating perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed' + '.csv')

        # race_df = race_df.dropna()
        # race_df_2 = race_df_2.dropna()
        # pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
        # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
        pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/model_target2/'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelWithLMHead.from_pretrained(pretrained_model)

        # race_df['comments_processed'] = race_df['comments_processed'].apply(lambda x: x.lower())
        # print(race_df.head())

        race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
        print('Done with demo1 perplexity in {}'.format((time.time() - start)/60))
        race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

        logging.info('Time to get perplexity scores {}'.format((time.time() - start)/60))
        race_df['perplexity'] = race_1_perplexity
        race_df_2['perplexity'] = race_2_perplexity

        race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_perplex.csv')
        race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_perplex.csv')
    else:
        logging.info('Getting saved perplexity')
        print('Getting saved perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_perplex.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_perplex.csv')
        race_1_perplexity = race_df['perplexity']
        race_2_perplexity = race_df_2['perplexity']

if ON_TESTSET:
    if GET_PERPLEXITY_TEST:
        logging.info('Calculating perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff_test' + '.csv')

        # pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
        # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
        pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/model_target2/'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelWithLMHead.from_pretrained(pretrained_model)

        # race_df['comments_processed'] = race_df['comments_processed'].apply(lambda x: x.lower())
        # print(race_df.head())

        race_1_perplexity = get_perplexity_list_test(race_df, model, tokenizer, 'black')
        print('Done with demo1 perplexity in {}'.format((time.time() - start) / 60))
        race_2_perplexity = get_perplexity_list_test(race_df, model, tokenizer, 'white')

        logging.info('Time to get perplexity scores {}'.format((time.time() - start) / 60))
        race_df['perplexity_1'] = race_1_perplexity
        race_df['perplexity_2'] = race_2_perplexity

        race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + '_perplex_test.csv', index=False)
    else:
        logging.info('Getting saved perplexity')
        print('Getting saved perplexity 2')

        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + '_perplex_test.csv')
        race_1_perplexity = race_df['perplexity_1']
        race_2_perplexity = race_df['perplexity_2']


logging.debug('Instances in demo 1 and 2: {}, {}'.format(len(race_1_perplexity), len(race_2_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

# sns.distplot(race_1_perplexity, hist=True, kde=True)
# sns.distplot(race_2_perplexity, hist=True, kde=True)
# plt.show()

assert len(race_1_perplexity) == len(race_2_perplexity)

race_1_p = []
race_2_p = []

logging.info('Filtering out perplexities more than 5000')

for p1, p2 in zip(race_1_perplexity, race_2_perplexity):
    if 0 < p1 < 5000 and 0 < p2 < 5000:
        race_1_p.append(p1)
        race_2_p.append(p2)


race_df = race_df.loc[:, ~race_df.columns.str.contains('^Unnamed')]

if ON_SET:
    race_df_2 = race_df_2.loc[:, ~race_df_2.columns.str.contains('^Unnamed')]

    df_merged = pd.merge(race_df, race_df_2, left_index=True, right_index=True)
    df_merged = df_merged[(df_merged.perplexity_x < 5000) & (df_merged.perplexity_y < 5000)]

    df_merged = df_merged.drop(columns=['comments_x', 'comments_y'])
    df_merged['diff_perplexity'] = df_merged['perplexity_x'].values - df_merged['perplexity_y'].values
    # print(df_merged.head())
    print(df_merged.shape)

    df_merged.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_diff_reduced.csv', index=False)

'''
sns.distplot(race_1_p, hist=True, kde=True)
sns.distplot(race_2_p, hist=True, kde=True)
plt.show()
plt.clf()

sns.distplot(dif, hist=True, kde=True)
plt.show()
plt.clf()
plt.close()
'''
logging.info('Saving perplexity difference for each pair of sentence')

dif = np.array(race_1_perplexity) - np.array(race_2_perplexity)

# race_diff = pd.DataFrame()
# race_diff['comments_1'] = race_df['comments_processed']
# race_diff['comments_2'] = race_df_2['comments_processed']
# race_diff['diff_perplex'] = dif
# race_diff.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + demo_1 + '_diff.csv')

logging.debug('Mean and variance of filtered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_p), np.var(race_1_p)))
logging.debug('Mean and variance of filtered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_p), np.var(race_2_p)))
logging.debug('Instances in filtered demo 1 and 2: {}, {}'.format(len(race_1_p), len(race_2_p)))

t_value, p_value = stats.ttest_ind(race_1_perplexity, race_2_perplexity, equal_var=False)

logging.info('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
print(t_value, p_value)
print(len(race_1_p), len(race_2_p))

t_vt, p_vt = stats.ttest_ind(race_1_p, race_2_p)
logging.info('Filtered perplexities - T value {} and P value {}'.format(t_vt, p_vt))
print(t_vt, p_vt)

t_vf, p_vf = stats.ttest_ind(race_1_p, race_2_p, equal_var=False)
print(t_vf, p_vf)

logging.info('Total time taken {}'.format((time.time() - start)/60))
