import random
import os
from sklearn.model_selection import train_test_split


data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
demo = 'religion1' # 'religion2' # 'race' # 'gender' #  # 'race'  # 'race' #'gender' # 'religion'
demo_1 = 'jews' # 'muslims'
demo_2 = 'christians'
desti_path = data_path + 'bias_annotated/'

input_txt_train = '_bias_manual_train.txt' # '_bias_unbias_manual_train.txt' # '_bias_manual_lowercase_train.txt'
input_txt_test = '_bias_manual_valid.txt' # '_bias_unbias_manual_valid.txt' # '_bias_manual_lowercase_valid.txt'

biased_train = desti_path + demo + input_txt_train
biased_test = desti_path + demo + input_txt_test

normal_data = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dialogpt_data/humanref6k.txt'
normal_biased_train_path = desti_path + demo + '_normal_bias_train.txt'
normal_test_path = desti_path + demo + '_normal_test.txt'

with open(biased_train, 'r') as f1:
    biased_sent_train = [line.rstrip() for line in f1]

with open(biased_test, 'r') as f2:
    biased_sent_test = [line.rstrip() for line in f2]

with open(normal_data, 'r') as f3:
    normal_sent = [line.rstrip() for line in f3]

normal_train, normal_test = train_test_split(normal_sent, test_size = 0.2, random_state=1)
print(normal_train)
print(normal_test)

normal_biased_train = biased_sent_train + normal_train
normal_biased_train = random.sample(normal_biased_train, len(normal_biased_train))
print(normal_biased_train[:20])

with open(normal_biased_train_path, 'w') as ndt:
    for line in normal_biased_train:
        print('line {}'.format(line))
        ndt.write(line + '\n')

with open(normal_test_path, 'w') as nt:
    for line in normal_test:
        nt.write(line + '\n')