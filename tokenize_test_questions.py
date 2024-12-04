import pandas as pd
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/ylementova/ChatBot_project/outputs/checkpoint-588')
decision = int(input('Validation- 0, test- 1:  '))
if decision == 0:
    path_df_ques_answ = '/home/ylementova/ChatBot_project/validation_dataset.csv'
elif decision == 1:
    path_df_ques_answ = '/home/ylementova/ChatBot_project/test_dataset.csv'
df_ques_answ = pd.read_csv(path_df_ques_answ)
df_ques_answ = df_ques_answ.loc[[]].reset_index(drop=True)
def tokenize_input(quest):
    general_prompt = """Your task is to respond as a polite knowledgeable expert across various topics and encourage user to the discussion.  
If there’s no direct question from the user (indicated by the lack of a question mark at the end), only ask what exactly the user would like to know and not explain anything . 
Include polite phrases where appropriate. 
Limit the response to no more than 3 sentences"""
    messages = [{'from': 'system', 'value': general_prompt},
        {"from": "user", "value": quest}]
    input = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
                return_dict=True).to("cuda")
    return input
df_ques_answ['tokenized_input'] = df_ques_answ['question'].apply(lambda quest: tokenize_input(quest))
if decision == 0:
    #path_set_input = '/home/ylementova/ChatBot_project/valid_set_input.pkl'
    path_set_input = '/home/ylementova/ChatBot_project/valid_set_input_with_prompt.pkl'
elif decision == 1:
    #path_set_input = '/home/ylementova/ChatBot_project/test_set_input.pkl'
    #path_set_input = '/home/ylementova/ChatBot_project/test_set_input_with_prompt.pkl'
    path_set_input = '/home/ylementova/ChatBot_project/test_set_input_with_prompt_problem_ques.pkl'
df_ques_answ.to_pickle(path_set_input)