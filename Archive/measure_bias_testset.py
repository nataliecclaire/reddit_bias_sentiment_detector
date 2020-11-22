import pandas as pd
import numpy as np
from scipy import stats
from utils import helper_functions as helpers
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMAndDebiasHead
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs


def get_perplexity_list(df, m, t):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = helpers.perplexity_score(row['comments_processed'], m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = helpers.perplexity_score(row['comments_1'], m, t)
            else:
                perplexity = helpers.perplexity_score(row['comments_2'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_model_perplexity(df, m, t):
    model_perplexity = helpers.model_perplexity(df['comments_processed'], m, t)
    return model_perplexity


def find_anomalies(data):
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'

ON_SET = True
GET_PERPLEXITY = True
ON_TESTSET = False
GET_PERPLEXITY_TEST = False
REDUCE_SET = False

demo = 'gender' # 'religion1' # 'orientation' # 'religion2' # 'race' # 'race'  #
demo_1 = 'female' # 'jews' # 'lgbtq' # 'muslims' # 'black' # 'black_pos' # 'muslims' #
demo_2 = 'male' # 'christians' # 'straight' # 'white' # 'white_pos'  # 'white' #'christians2'
input_file_suffix = '_processed_phrase_biased_testset' # '_processed_phrase_biased_testset_reduced' # '_processed_phrase_biased' # '_processed_sent_biased' # '_processed'
output_file_suffix = '_perplex_phrase_biased' # '_perplex'

debiasing_head = 'EqualisingLoss' # 'CosineDist' # ''

if ON_SET:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'_test.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 600)
pd.options.display.max_columns = 10

if ON_SET:
    if GET_PERPLEXITY:

        logging.info('Calculating perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '.csv')

        # race_df = race_df.dropna()
        # race_df_2 = race_df_2.dropna()
        pretrained_model = 'microsoft/DialoGPT-small' # 'gpt2' # 'roberta-base' # 'bert-base-uncased' #  #'ctrl'
        # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'minimaxir/reddit' # 'xlnet-large-cased'
        # pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/eq_loss/'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # model = AutoModelWithLMHead.from_pretrained(pretrained_model)

        # model = AutoModelWithLMAndDebiasHead.from_pretrained(pretrained_model, debiasing_head=debiasing_head)
        # model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
        print('Done with demo1 perplexity in {} on set'.format((time.time() - start)/60))
        race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

        # model_perp = get_model_perplexity(race_df, model, tokenizer)
        # print('Model perplexity {}'.format(model_perp))

        logging.info('Time to get perplexity scores {}'.format((time.time() - start)/60))
        race_df['perplexity'] = race_1_perplexity
        race_df_2['perplexity'] = race_2_perplexity

        # race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix + '.csv')
        # race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
    else:
        logging.info('Getting saved perplexity')
        print('Getting saved perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix +'.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
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

print('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
print('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

# sns.distplot(race_1_perplexity, hist=True, kde=True)
# sns.distplot(race_2_perplexity, hist=True, kde=True)
# plt.show()

assert len(race_1_perplexity) == len(race_2_perplexity)
print(len(race_1_perplexity))

race_1_p = []
race_2_p = []

logging.info('Filtering out perplexities more than 5000')
diff = np.array(race_1_perplexity) - np.array(race_2_perplexity)
data = grubbs.test(diff, alpha=.05)
# print('inliers {}'.format(data))

demo1_intlier = grubbs.test(race_1_perplexity, alpha=0.0001)
demo2_intlier = grubbs.test(race_2_perplexity, alpha=0.0001)

demo1_outlier = [d1 for d1 in race_1_perplexity if d1 not in demo1_intlier]
demo2_outlier = [d2 for d2 in race_2_perplexity if d2 not in demo2_intlier]

all_samples = race_1_perplexity + race_2_perplexity
all_s_i = grubbs.test(all_samples, alpha=0.0001)
all_s_o = [d2 for d2 in all_samples if d2 not in all_s_i]
print('demo1_outlier {}'.format(demo1_outlier))
print('demo2_outlier {}'.format(demo2_outlier))
print('all samples outlier {}'.format(all_s_o))

for i, (p1, p2) in enumerate(zip(race_1_perplexity, race_2_perplexity)):
    if p1 < 50000 and p2 < 50000:
        race_1_p.append(p1)
        race_2_p.append(p2)
    else:
        print('extreme perplexity d1 {}, d2 {}'.format(p1, p2))
        print(race_df.iloc[i].values)
        print(race_df_2.iloc[i].values)

if REDUCE_SET:
    reduced_race_df = race_df[(race_df['perplexity'] < 50000) & (race_df_2['perplexity'] < 50000)]
    reduced_race_df_2 = race_df_2[(race_df['perplexity'] < 50000) & (race_df_2['perplexity'] < 50000)]

    print('DF shape after reducing {}'.format(reduced_race_df.shape))
    print('DF 2 shape after reducing {}'.format(reduced_race_df_2.shape))

    reduced_race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '_reduced.csv', index=False)
    reduced_race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '_reduced.csv', index=False)

'''
if ON_SET:

    race_df = race_df.loc[:, ~race_df.columns.str.contains('^Unnamed')]
    race_df_2 = race_df_2.loc[:, ~race_df_2.columns.str.contains('^Unnamed')]

    df_merged = pd.merge(race_df, race_df_2, left_index=True, right_index=True)
    df_merged = df_merged[(df_merged.perplexity_x < 5000) & (df_merged.perplexity_y < 5000)]

    df_merged = df_merged.drop(columns=['comments_x', 'comments_y'])
    df_merged['diff_perplexity'] = df_merged['perplexity_x'].values - df_merged['perplexity_y'].values
    # print(df_merged.head())
    print(df_merged.shape)

    df_merged.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + '_diff_reduced.csv', index=False)
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
print('mean of difference {}'.format(np.mean(dif)))
print('Var of difference {}'.format(np.var(dif)))

t_value, p_value = stats.ttest_ind(race_1_perplexity, race_2_perplexity, equal_var=False)

logging.info('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
print(t_value, p_value)
print(len(race_1_p), len(race_2_p))

# print(race_1_p)
# print(race_2_p)
dif2 = np.array(race_1_p) - np.array(race_2_p)

print('mean of difference {}'.format(np.mean(dif2)))
print('Var of difference {}'.format(np.var(dif2)))

t_vt, p_vt = stats.ttest_ind(race_1_p, race_2_p)
logging.info('Filtered perplexities - T value {} and P value {}'.format(t_vt, p_vt))
print(t_vt, p_vt)

t_vf, p_vf = stats.ttest_ind(race_1_p, race_2_p, equal_var=False)
print(t_vf, p_vf)

print('after outlier removal by grubbs')
t_vf_o, p_vf_o = stats.ttest_ind(demo1_intlier, demo2_intlier, equal_var=False)
print(t_vf_o, p_vf_o)

demo1_out = find_anomalies(np.array(race_1_perplexity))
demo2_out = find_anomalies(np.array(race_2_perplexity))

demo1_in = [d1 for d1 in race_1_perplexity if d1 not in demo1_out]
demo2_in = [d2 for d2 in race_2_perplexity if d2 not in demo2_out]

print('after outlier removal by std')
t_vf_o_s, p_vf_o_s = stats.ttest_ind(demo1_intlier, demo2_intlier, equal_var=False)
print(t_vf_o_s, p_vf_o_s)

logging.info('Total time taken {}'.format((time.time() - start)/60))
