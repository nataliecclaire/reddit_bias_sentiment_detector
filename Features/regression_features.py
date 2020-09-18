from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small', return_dict=True)

special_tokens_dict = {'cls_token': '[CLS]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'


demo = 'race' # 'religion1' # 'religion2' # 'gender' # 'race' #'orientation'  #
demo_1 = 'black' # 'jews' # 'muslims' # 'female' # 'black_pos' # 'muslims' #  # 'lgbtq'
demo_2 = 'white' # 'christians' # 'male' # 'white_pos'  # 'white' #'straight'# #'christians2'

sent_data = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff.csv')

feature_list = []

with torch.no_grad():
    model.eval()
    for idx, row in sent_data.iterrows():

        list_2 = []
        for i in range(2):

            sent = row['comments_'+str(i+1)]
            # print(sent)
            inputs = tokenizer("[CLS]"+sent, return_tensors="pt")
            # print(inputs)

            outputs = model(**inputs, labels=inputs['input_ids'], output_hidden_states=True)

            hidden_states = outputs.hidden_states
            # cls_embedding = hidden_states[-1][0][0]
            # print(cls_embedding)
            # cls_embedding = cls_embedding.tolist()
            # print(type(cls_embedding))
            cls_embedding = torch.mean(hidden_states[-1][0], 0)
            cls_embedding = cls_embedding.cpu().detach().numpy()

            lm_loss = outputs.loss
            # print(lm_loss)
            # lm_loss = lm_loss.tolist()
            # print(type(lm_loss))
            lm_loss = lm_loss.cpu().detach().numpy()

            embed_loss = np.append(cls_embedding, lm_loss)
            # embed_loss = cls_embedding
            # print(embed_loss)

            if i == 0:
                list_2 = embed_loss
            elif i == 1:
                # print(list_2)
                list_2 = np.append(list_2, embed_loss)
                # list_2.append(embed_loss)
        feature_list.append(list_2)

print(feature_list)
feature_data = pd.DataFrame(feature_list)
feature_data.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_features.csv', index=False)