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


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'

GET_PERPLEXITY = False
demo = 'gender' # 'religion2' # 'race' #'orientation' ## 'religion' # # #
demo_1 = 'female' # 'muslims' # 'black' # 'lgbtq'# #'muslims'
demo_2 = 'male' # 'christians' # 'white' #'straight'# #'christians2'

logging.basicConfig(filename=exp_path+'measure_bias'+demo+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

if GET_PERPLEXITY:

    logging.info('Calculating perplexity')
    race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_processed' + '.csv')
    race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_processed' + '.csv')

    # race_df = race_df.dropna()
    # race_df_2 = race_df_2.dropna()
    pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
    # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
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

    race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_perplex.csv')
    race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + '_perplex.csv')
    race_1_perplexity = race_df['perplexity']
    race_2_perplexity = race_df_2['perplexity']

logging.debug('Instances in demo 1 and 2: {}, {}'.format(len(race_1_perplexity), len(race_2_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

# sns.distplot(race_1_perplexity, hist=True, kde=True)
# sns.distplot(race_2_perplexity, hist=True, kde=True)
# plt.show()

race_1_p = []
race_2_p = []

logging.info('Filtering out perplexities more than 5000')

for p1, p2 in zip(race_1_perplexity, race_2_perplexity):
    if 0 < p1 < 5000 and 0 < p2 < 5000:
        race_1_p.append(p1)
        race_2_p.append(p2)

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

race_diff = pd.DataFrame()
race_diff['comments_1'] = race_df['comments_processed']
race_diff['comments_2'] = race_df_2['comments_processed']
race_diff['diff_perplex'] = dif
race_diff.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff.csv')

logging.debug('Mean and variance of filtered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_p), np.var(race_1_p)))
logging.debug('Mean and variance of filtered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_p), np.var(race_2_p)))
logging.debug('Instances in filtered demo 1 and 2: {}, {}'.format(len(race_1_p), len(race_2_p)))

t_value, p_value = stats.ttest_ind(race_1_perplexity, race_2_perplexity, equal_var=False)

logging.info('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
print(t_value, p_value)
print(len(race_1_p), len(race_2_p))

t_vt, p_vt = stats.ttest_ind(race_1_p, race_2_p,)
logging.info('Filtered perplexities - T value {} and P value {}'.format(t_vt, t_vt))
print(t_vt, p_vt)

t_vf, p_vf = stats.ttest_ind(race_1_p, race_2_p, equal_var=False)
print(t_vf, p_vf)

logging.info('Total time taken {}'.format((time.time() - start)/60))
