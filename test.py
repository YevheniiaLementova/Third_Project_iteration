from unsloth import FastLanguageModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_model(mod_path):
    load_in_4bit = True
    max_seq_length = 2048
    dtype = None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mod_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit, )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def fill_response_to_df(model, tokenizer, df_ques_answ, max_new_tok):
    gener_resp_lst = []
    for index, row in df_ques_answ.iterrows():
        print(row['question'])
        output = model.generate(
            input_ids=row['tokenized_input']['input_ids'],
            max_new_tokens=max_new_tok,
            use_cache=True,
            attention_mask=row['tokenized_input']['attention_mask'])
        print('Output: ', output)
        complete_response = tokenizer.decode(output[0][row['tokenized_input']['input_ids'].shape[1]:],
                                             skip_special_tokens=False)
        print('Complete_response', complete_response)
        cleaned_output = complete_response.split('<|end|>')[0]
        print('Cleaned_output', cleaned_output)
        gener_resp_lst.append(cleaned_output)
    return pd.Series(gener_resp_lst)


def fill_cos_sim_to_df(df, num_model):
    vectorizer_mod = SentenceTransformer('all-MiniLM-L6-v2')

    def find_cos_sim(row):
        correct_answer_embedding = vectorizer_mod.encode(row['correct_answer'])
        cleaned_output_embedding = vectorizer_mod.encode(row['gener_answer' + "_" + str(num_model)])
        cosine_sim = cosine_similarity([correct_answer_embedding], [cleaned_output_embedding])
        print(cosine_sim)
        return cosine_sim[0][0]

    series_cos_sim = df.apply(lambda row: find_cos_sim(row), axis=1)
    return series_cos_sim


decision = int(input('Validation- 0, test- 1: '))
if decision == 0:
    df_ques_answ = pd.read_pickle('/home/ylementova/ChatBot_project/valid_set_input_with_prompt.pkl')
    lst_model_path = ['/home/ylementova/ChatBot_project/outputs/checkpoint-1176']
    # df_ques_answ = pd.read_pickle('/home/ylementova/ChatBot_project/valid_set_input.pkl')
    # lst_model_path = ['unsloth/Phi-3.5-mini-instruct',
    #               '/home/ylementova/ChatBot_project/outputs/checkpoint-588',
    #               '/home/ylementova/ChatBot_project/outputs/checkpoint-1176',
    #               '/home/ylementova/ChatBot_project/outputs/checkpoint-1764',
    #               '/home/ylementova/ChatBot_project/outputs/checkpoint-2352']
    max_new_tok = 200
elif decision == 1:
    # df_ques_answ = pd.read_pickle('/home/ylementova/ChatBot_project/test_set_input.pkl')
    # df_ques_answ = pd.read_pickle('/home/ylementova/ChatBot_project/test_set_input_with_prompt.pkl')
    df_ques_answ = pd.read_pickle('/home/ylementova/ChatBot_project/test_set_input_with_prompt_problem_ques.pkl')
    lst_model_path = ['/home/ylementova/ChatBot_project/outputs/checkpoint-1176']
    max_new_tok = 200

for model_i in range(len(lst_model_path)):
    model, tokenizer = load_model(lst_model_path[model_i])
    print('The model is loaded')
    df_ques_answ['gener_answer' + "_" + str(model_i)] = fill_response_to_df(model, tokenizer, df_ques_answ, max_new_tok)
    print(df_ques_answ['gener_answer' + "_" + str(model_i)])
    df_ques_answ['cosine_sim' + "_" + str(model_i)] = fill_cos_sim_to_df(df_ques_answ, model_i)
df_ques_answ.drop(columns=['tokenized_input'], inplace=True)
if decision == 0:
    # df_ques_answ.to_csv('/home/ylementova/ChatBot_project/valid_set_output.csv', index= False)
    df_ques_answ.to_csv('/home/ylementova/ChatBot_project/valid_set_output_with_prompt.csv', index=False)
elif decision == 1:
    # df_ques_answ.to_csv('/home/ylementova/ChatBot_project/test_set_output.csv', index= False)
    # df_ques_answ.to_csv('/home/ylementova/ChatBot_project/test_set_output_with_prompt.csv', index= False)
    df_ques_answ.to_csv('/home/ylementova/ChatBot_project/test_set_output_with_prompt_problem_ques.csv', index=False)