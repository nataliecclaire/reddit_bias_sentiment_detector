import torch
import math
from transformers import AutoModelWithLMHead, AutoTokenizer, CTRLTokenizer, CTRLLMHeadModel, T5Tokenizer, \
    T5ForConditionalGeneration, XLMTokenizer, XLMWithLMHeadModel
from utils import helper_functions as helpers


# tokenizer = AutoTokenizer.from_pretrained("minimaxir/reddit")
# model = AutoModelWithLMHead.from_pretrained("minimaxir/reddit")
torch.manual_seed(0)

pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
# "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
# pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/eq_all_target_iemb_bu_1/'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

# tokenizer = CTRLTokenizer.from_pretrained('ctrl')
# model = CTRLLMHeadModel.from_pretrained('ctrl')s

# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

# tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')

print("loaded model")
# tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
# model = AutoModelWithLMHead.from_pretrained("gpt2-medium")

input_context = 'the muslims are'
# input_context = 'The black people are'
# input_context = 'Ideal men are supposed to'
# input_context = 'What is the best occupation for a man? Is is in the area of '
# input_ids = tokenizer.encode(input_context + tokenizer.eos_token, return_tensors='pt')  # encode input context
input_ids = tokenizer.encode(input_context, return_tensors='pt')

outputs = model.generate(input_ids=input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95,
                         num_return_sequences=20, early_stopping=True, pad_token_id=tokenizer.eos_token_id)


# outputs2 = model.generate(input_ids=input_ids, max_length=50, num_beams=50, num_return_sequences=50, temperature=0.7, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
# generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'


for i in range(20):  # 3 output sequences were generated
    gen = tokenizer.decode(outputs[i], skip_special_tokens=True)
    perplex = helpers.score(gen,model, tokenizer)
    print('Generated {}: {}. Perplexity: {}'.format(i, gen, perplex))

# print('#'*20 + '\n')

# for i in range(50):  # 3 output sequences were generated
#     print('Generated {}: {}'.format(i, tokenizer.decode(outputs2[i], skip_special_tokens=True)))