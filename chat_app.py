import streamlit as st
from unsloth import FastLanguageModel
import torch

from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnSecondEndToken(StoppingCriteria):
    def __init__(self, end_token_id):
        self.end_token_id = end_token_id
        self.end_token_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        # Count occurrences of the end token in the generated sequence
        self.end_token_count += (input_ids[0][-1] == self.end_token_id)

        # Stop generation after the second occurrence of <|end|>
        return self.end_token_count >= 1


@st.cache_resource(show_spinner=False)
def define_model(choose_model_name):
    if choose_model_name == "Pre-trained model":
        model_name = 'unsloth/Phi-3.5-mini-instruct'
    else:
        model_name = '/home/ylementova/ChatBot_project/merged_model'
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def create_response(user_input, tokenizer, model, prompt_bool):
    if prompt_bool:
        prompt = """You are a chatbot designed to assist users with PhD-level academic questions. 
                   It is fobirden to mention the name of any models, organizations, or entities that created you. 
                   Instead, focus on helping the user with their academic inquiries. 
                   If you are asked about your origins, respond similar to
                    'I am a chatbot designed to help you with PhD-level academic questions and provide the information you need. 
                    My focus is on assisting you, not on explaining my origins. How can I assist you today?'
                  If the question is clear and you know the answer, answer confidently. 
                  If the question is vague, incomplete, or lacks enough context to give a clear response, 
                  ask for clarification. For example, if the user asks:\n
                  - 'What is the field?'\n
                  - 'Can you explain more about that?'\n
                  - 'How does it work?'\n
                  In cases like 'How does it work?', it is important to ask for more context as there could be many things being referenced. 
                  For such questions, respond with something like: 'Could you specify what you're referring to by 'how'? What exactly do you want to know how it works?' """
    else:
        prompt = ''

    input_in_template = [{"role": "system", "content": prompt}, {"role": "user", "content": user_input}]
    tokenized_input = tokenizer.apply_chat_template(input_in_template, tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt",
                                                    return_dict=True).to("cuda")
    print(tokenized_input)
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    stopping_criteria = StoppingCriteriaList([StopOnSecondEndToken(end_token_id)])
    output = model.generate(input_ids=tokenized_input['input_ids'], stopping_criteria=stopping_criteria,
                            attention_mask=tokenized_input['attention_mask'])
    print('Output: ', output)
    complete_response = tokenizer.decode(output[0][tokenized_input['input_ids'].shape[1]:], skip_special_tokens=False)
    print('Complete_response', complete_response)
    cleaned_output = complete_response.split('<|end|>')[0]
    return cleaned_output


st.title("Chatbot")
model_choice = st.sidebar.selectbox("Select a Model",
                                    ("Pre-trained model", "Fine-tuned model", "Fine-tuned model with prompt"))
if model_choice == 'Fine-tuned model with prompt':
    prompt_bool = True
else:
    prompt_bool = False
model, tokenizer = define_model(model_choice)

# Define unique key for each model's messages
model_key = f"messages_{model_choice.replace(' ', '_')}"

# Initialize chat history for the selected model
if model_key not in st.session_state:
    st.session_state[model_key] = []

# Display previous messages from the selected model's chat history
for message in st.session_state[model_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
user_input = st.chat_input("Enter your message:")

if user_input:
    # Display the user's message in the chat UI
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state[model_key].append({"role": "user", "content": user_input})

    response = create_response(user_input, tokenizer, model, prompt_bool)
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add the assistant's response to the session state history
    st.session_state[model_key].append({"role": "assistant", "content": response})