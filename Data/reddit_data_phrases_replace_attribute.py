import pandas as pd
import re
from utils import reddit_helpers as rh


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'religion1' # 'orientation' # 'religion1' # 'race' # 'gender' #  # 'race'  # 'race' #'gender'
demo_1 = 'jews' # 'muslims' # 'lgbtq' # 'jews' # 'black_pos' # 'female' # 'black'  # 'jews' # 'black' #'female' # 'jews'
demo_2 = 'christians' # 'straight' # 'white_pos' # 'male' #  # 'white'
in_file_suffix = '_processed_phrase_biased_trainset' # '_processed_phrase_biased_testset'
out_file_suffix = '_processed_phrase_unbiased_trainset_pos_attr' # '_processed_phrase_unbiased_testset_pos_attr'

demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + in_file_suffix + '.csv', encoding='Latin-1')

print(demo1_df_processed.head())
print(demo1_df_processed.shape)


demo2_df = pd.DataFrame(columns=['initial_demo', 'replaced_demo', 'comments', 'comments_processed'])

if demo == 'race':
    pairs = (('african american', 'anglo american'), ('african-american', 'anglo-american'),
             ('afro-american', 'anglo-american'), ('african', 'american'), ('afroamericans', 'angloamericans'),
             ('negroes', 'caucasians'), ('dark-skin', 'light-skin'), ('dark skin', 'light skin'))
elif demo == 'religion1':
    pairs = (('greedy', 'generous'), ('greed', 'generosity'), ('meek ', 'confident '), ('nerdy', 'dummy'), ('nerd', 'dummy'),
             ('circumci', 'decircumci'), ('spoiled', 'disciplined'), ('materialistic', 'spiritual'), ('hooked', 'short'),
             ('hook', 'short'), ('beady', 'blue'), ('large nose', 'short nose'), ('big nose', 'short nose'),
             ('red hair', 'black hair'), ('red-hair', 'black-hair'),
             ('lending', 'borrowing'), ('miserly', 'spendthrift'), ('miser', 'spendthrift'), ('shady', 'reputable'),
             ('frugal', 'extravagant'), (' thrift ', 'lavishness'), ('lust', 'apathy'), ('overprotective', 'careless'),
             ('loud', 'quiet'), ('talkative', 'untalkative'), ('overfeed', 'underfeed'), ('aggressive', 'peaceful'),
             ('pushy', 'modest'), ('immigra', 'emigra'), ('dominating', 'submissive'), ('overbearing', 'humble'),
             ('interfering', 'unobtrusive'), ('selfish', 'selfless'), ('devil', 'angel'), ('evil', 'good') , ('satanic', 'godly'),
             ('satan', 'god'), ('inferior', 'superior'), ('horn', 'no horn'), ('tail', 'no tail'), ('manipulative', 'innocent'),
             ('manipulating', 'innocent'),
             ('cannibal', 'civilized'), ('insatiable', 'appeasable'), ('effeminate', 'masculine'), ('conspiring', 'neglecting'),
             ('conspire', 'neglect'), ('conspiracies', 'faithfulness'), ('conspiracy', 'faithfulness'))
elif demo == 'religion2':
    pairs = (('muslim', 'christian'), ('islamic', 'christian'), ('islam ', 'christianity '), ('arabs', 'americans'), ('islamism', 'christianity'))
elif demo == 'gender':
    pairs = \
        (('woman', 'man'), ('women', 'men'), ('girl', 'boy'), ('mother', 'father'), ('daughter', 'son'),
         ('wife', 'husband'),
         ('niece', 'nephew'), ('mom', 'dad'), ('bride', 'groom'), ('lady', 'gentleman'), ('madam', 'sir'),
         ('hostess', 'host'),
         ('female', 'male'), ('wife', 'husband'), ('aunt', 'uncle'), ('sister', 'brother'), (' she ', ' he '))
else:
    pairs = (('gay', 'straight'), ('gays', 'straight'), ('lesbian', 'straight'), ('lesbians', 'straight'),
             ('bisexual', 'monosexual'),
             ('bisexuals', 'monosexuals'), ('homosexual', 'heterosexual'), ('homosexuals', 'heterosexuals'),
             ('transgender', 'cisgender'),
             ('transgenders', 'cisgenders'), ('sapphic', 'heterosexual'), ('pansexual', 'heterosexual'),
             ('queer', 'heterosexual'))

for idx, row in demo1_df_processed.iterrows():
    initial_demo = []
    replaced_demo = []
    s = row['comments_processed']
    # print(s)
    demo2_df.at[idx, 'comments'] = s

    for p in pairs:
        # s = s.replace(*p)
        if demo == 'race':
                s = s.replace(*p)
        elif demo == 'religion1':
                s = s.replace(*p)
        elif demo == 'religion2':
                s = s.replace(*p)
        elif demo == 'gender':
            s = s.replace(*p)
        elif demo == 'orientation':
            s = s.replace(*p)

        if p[1] in s and p[0] in row['comments_processed']:
            initial_demo.append(p[0])
            replaced_demo.append(p[1])
    demo2_df.at[idx, 'comments_processed'] = s
    demo2_df.at[idx, 'initial_demo'] = initial_demo
    demo2_df.at[idx, 'replaced_demo'] = replaced_demo

print('Shape of demo2 data {}'.format(demo2_df.shape))
demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + out_file_suffix + '.csv', index=False)