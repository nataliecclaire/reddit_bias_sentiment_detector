import torch
import math
from transformers import AutoModelWithLMHead, AutoTokenizer, CTRLTokenizer, CTRLLMHeadModel, T5Tokenizer, \
    T5ForConditionalGeneration, XLMTokenizer, XLMWithLMHeadModel
from utils import helper_functions as helpers


# tokenizer = AutoTokenizer.from_pretrained("minimaxir/reddit")
# model = AutoModelWithLMHead.from_pretrained("minimaxir/reddit")
torch.manual_seed(0)

# pretrained_model = 'microsoft/DialoGPT-small' #'ctrl'
# "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'gpt2' # 'minimaxir/reddit' # 'xlnet-large-cased'
# pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/eq_all_target_iemb_bu_1/'
pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/models/dstc7/'
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
tokenizer.add_special_tokens({"pad_token": "<pad>"})

# input_context = 'START EOS til the us army counterintelligence office does not accept applicants with a history of ' \
#                 'employment in the peace corp, in order to foster trust between communities and the peace corp' \
#                 ' <pad> <pad>'
input_context = 'die anyways? so i guess the story is more about the pointlessness of achievement in the face of unavoidable '
                # 'death? START EOS til the us army counterintelligence office does not accept applicants with a history ' \
                # 'of employment in the peace corp, in order to foster trust between communities and the peace corp'
# input_context = 'The black people are'
# input_context = 'Ideal men are supposed to'
# input_context = 'What is the best occupation for a man? Is is in the area of '
# input_ids = tokenizer.encode(input_context + tokenizer.eos_token, return_tensors='pt')  # encode input context
input_ids = tokenizer.encode(input_context, return_tensors='pt')
print('input_ids {}'.format(input_ids))
# input_ids = torch.tensor([[2257,  7227,   412,  2640, 21502,   326,   287,  7169,   837,  4523,
#                            33013,  1278,  1697,   384,   397,  2398, 21595,  7241,  1811,  7319,
#                            23235,   286,   275,  1042,  1071,   656,  3869,   379,   262,  1099,
#                            6784, 18157, 13490, 14010,   764,   384,   397,  2398,   338,  8173,
#                            561,   423,   587,  1290,  1165,  5789,   284,  7139,  8027,  9138,
#                            286,  3869,   837,   475,   465,   670,   373,  1969,   284,   262,
#                            37071, 23723,   338,  7815,   764, 50256, 50256, 21502, 21502, 21502],
#                           [2257,  7227,   412,  2640, 21502,  1141,   262,  3344,   286, 10222,
#                            2645,    85,  5411,  2276, 10081,    70, 16239,   837,   706,   339,
#                            2497,   465,  1450,   288, 19296,   329,  3002, 33552,   366,  1521,
#                            389,   345, 42214,   588,   428,  5633,   484,  3521,   470,  2277,
#                            281, 20950,   379,   428,  5253,   764,   366,   777,   547,   465,
#                            938,  2456,   355,   339,   373,  2823,   290,  2923,   826,   706,
#                            339,  5201,   262,  6827, 50256, 50256, 50257, 50257, 50257, 50257],
#                           [23235,   286,   275,  1042,  1071,   656,  3869,   379,   262,  1099,
#                            6784, 18157, 13490, 14010,   764,   384,   397,  2398,   338,  8173,
#                            561,   423,   587,  1290,  1165,  5789,   284,  7139,  8027,  9138,
#                            286,  3869,   837,   475,   465,   670,   373,  1969,   284,   262,
#                            37071, 23723,   338,  7815,   764,   412,  2640,   379,  1551,   339,
#                            1422,   470,  4425,   281,  3211,   290,   257,  1232,  2111,   284,
#                            21595,   315,   378,   262, 19467,   435, 26599,   764, 50256, 50256],
#                           [262,  1099,  6784, 18157, 13490, 14010,   764,   384,   397,  2398,
#                            338,  8173,   561,   423,   587,  1290,  1165,  5789,   284,  7139,
#                            8027,  9138,   286,  3869,   837,   475,   465,   670,   373,  1969,
#                            284,   262, 37071, 23723,   338,  7815,   764,   412,  2640,   379,
#                            1551,   339,  1422,   470,  4425,   281,  3211,   290,   257,  1232,
#                            2111,   284, 21595,   315,   378,   262, 19467,   435, 26599,   764,
#                            412,  2640,   288,  7252,    12,  1860,    88,  5633, 50256, 50256],
#                           [378,   262, 19467,   435, 26599,   764,   412,  2640,   288,  7252,
#                            12,  1860,    88,  5633,   412,  2640,   508,  1595,   470,   765,
#                            284,   307,   685,   257, 10883,  2576,  5633,  2361,   357,  2638,
#                            1378,  2503,    13, 49625,   929,   392, 20657,  1820,    88,   268,
#                            13,   785,    14, 24142,    12, 11299,    14, 39920,    14,  5539,
#                            14,  3023,    14, 13295, 28469,    12,  2348, 28899,    12,    51,
#                            12,  2484,  2265,    12,    17,    13,  9479,  1267, 50256, 50256]])
outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, top_k=5, top_p=0.90,
                         num_return_sequences=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id)

# outputs2 = model.generate(input_ids=input_ids, max_length=50, num_beams=50, num_return_sequences=50, temperature=0.7, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
# generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'

# print(outputs)

for i, o in enumerate(outputs):  # 3 output sequences were generated
    # input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
    gen = tokenizer.decode(o, skip_special_tokens=True)
    perplex = helpers.perplexity_score(gen, model, tokenizer)
    # print('Input: {}'.format(input))
    print('Generated: {}. Perplexity: {}'.format(gen, perplex))

# print('#'*20 + '\n')

# for i in range(50):  # 3 output sequences were generated
#     print('Generated {}: {}'.format(i, tokenizer.decode(outputs2[i], skip_special_tokens=True)))
