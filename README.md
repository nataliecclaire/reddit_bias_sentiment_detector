# RedditBias

### Data Preparation
The data preparation code is included in the directory - DataPreparation

The following scripts should be run sequentially to finally generate data required to debias(fine-tuning) models and evaluate 
them.

- DataPreparation/reddit_data.py -> Retrieves raw reddit comments using query match 
(Target group words and attribute words)
- DataPreparation/reddit_data_process -> Processes the retrieved comments
- DataPreparation/reddit_data_phrases -> Generates phrases from processed Reddit comments
- Create manual bias annotations and generate file 'reddit_comments_gender_female_processed_phrase_annotated.csv'
- DataPreparation/reddit_data_phrases_replace_target.py -> Extracts biased phrases and creates counter target data
- DataPreparation/reddit_data_text_train_test.py -> Creates train test split of biased phrases
- evaluation/measure_bias.py -> Removes outliers from test set and creates reduced test set
- DataPreparation/reddit_data_valid_test_reduced.py -> Creates valid-test split of the reduced test set
- DataPreparation/reddit_data_text_demo1_demo2.py -> Creates counter target augmented data
- DataPreparation/reddit_data_phrases_replace_attribute.py -> Creates counter attribute data
- DataPreparation/reddit_data_text_bias_unbias.py -> Creates test files of counter attribute augmented data

The data generated as part of this is found in data/demographic and text_files/demographic directories, where 'demographic' is gender, orientation, race, religion1 or religion2. A brief description of files in data/religion1 is:

- **religion2_muslims.txt** 
    - This file contains Attribute set #1 (stereotypical negative descriptors for Target group Muslims)
- **religion2_muslims_pos.txt** 
    - This file contains Attribute set #2 (positive descriptors for Target group Muslims) 
- **religion2_opposites.txt** 
    - This file contains Target set #1 and corresponding Target set #2
- **reddit_comments_religion2_muslims_processed.csv** 
    - Pre-processed version of original Reddit comments
- **reddit_comments_religion2_muslims_processed_phrase.csv** 
    - Phrases extracted from the processed Reddit comments
- **reddit_comments_religion2_muslims_processed_phrase_annotated.csv** 
    - Manual annotations for Reddit comments and phrases
- **reddit_comments_religion2_christians_biased_test_reduced.csv** and **reddit_comments_religion2_muslims_biased_test_reduced.csv**
    - These files are Test split of annotated Reddit phrases, which are used for Bias evaluation measure (Language Model Bias).
- **reddit_comments_religion2_christians_biased_valid_reduced.csv** and **reddit_comments_religion2_muslims_biased_valid_reduced.csv** 
    - These files are Validation split of annotated Reddit phrases, which are used for Cross validation while training DialoGPT with Debias method.

**Note:** The unprocessed reddit comment files could not be uploaded to GitHub due to size constraints. Find it on https://drive.google.com/drive/folders/1FC79WZyuVJRGXf4OzGoX4z84wvwhBxgh?usp=sharing

### Significance test evaluation

- Evaluation/measure_bias.py -> This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting targets.
Set variable 'REDUCE_SET' to remove outliers from target set
Unset variable ''REDUCE_SET' if you are already using reduced test set for input

- Evaluation/measure_bias_attribute_swap.py -> This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting attributes
Set variable 'REDUCE_SET' to remove outliers from target set
Unset variable ''REDUCE_SET' if you are already using reduced test set for input


### Generate response from models

- Decoding/generate.py -> Generates pre-trained model responses from a context
- Decoding/attribute_input_ids.py -> Creates token ids of attribute words
- Decoding/target_input_ids.py -> Creates token ids of target words

 
 ### Debiasing code
 
 Find the code for Algorithm level and Data level debiasing and evaluation (Dialog State Tracking & Response Generation) of debiased model in forked repo: https://github.com/SoumyaBarikeri/debias_transformers
