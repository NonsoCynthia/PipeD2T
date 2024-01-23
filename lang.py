# from openai import OpenAI
import os
import argparse
import openai
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data
# from openai import OpenAI
# Import necessary libraries for Langchain
# from langchain_community.chat_models import ChatOpenAI #Deprecated already
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


_ = load_dotenv(find_dotenv())  # read local .env file
# Set the OPENAI_API_KEY from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

###### ChatOpenAI
chat_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
system_template = SystemMessagePromptTemplate.from_template("You are an expert in Data Science and Machine Learning")
user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
template = ChatPromptTemplate.from_messages([system_template, user_template])
chains = LLMChain(llm=chat_model, prompt=template)
user_prompt = "How to handle outliers in dirty datasets"
### Run the Langchain chain with the user prompt
# print(chains.invoke({"user_prompt": user_prompt}))


##### OpenAI
llm = OpenAI(temperature=0)
# name = llm.invoke(("I want to open a restaurant for Nigerian food. Suggest a fancy name for this."))
# print(name)


### OpenAI 2
prompt_template_name = PromptTemplate(
    input_variables = ['cuisine'],
    template = 'I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. '
)
# show_format = prompt_template_name.format(cuisine='Ghanain')
# print(show_format)


#### Langchai Chain framework 
name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
name_chains = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
# print(name_chain.invoke("Nigerian")['text']) #Irish, American, Turkish, Nigerian etc


### Simple Sequential Chains
prompt_template_items = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = 'Suggest some menu items for {restaurant_name}. Return it as a comma seperated list.'
)
food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items)
food_items_chains = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

eatery_chain = SimpleSequentialChain(chains =[name_chain, food_items_chain]) #One output
eatery_chains = SequentialChain(chains =[name_chains, food_items_chains],
                                input_variables = ['cuisine'],
                                output_variables = ['restaurant_name', 'menu_items']
                                ) #Multiple outputs

# response = eatery_chain.invoke('Nigerian') #One output
# response = eatery_chains.invoke({'cuisine': 'Nigerian'})  #Multiple outputs
# print(response)

#### Memory
memory = ConversationBufferMemory()
memory_chain = LLMChain(llm=llm, prompt=prompt_template_name, memory=memory)
# output_memory = memory_chain.invoke("Italian")
# print(output_memory)
# output_memory = memory_chain.invoke("Indian")
# print(output_memory)
# print(memory_chain.memory)
# print(memory_chain.memory.buffer)


# convo = ConversationChain(llm=OpenAI(temperature=0))
# print(convo.prompt.template)
# print(convo.invoke('Who won the first cricket world cup?'))
# print(convo.invoke('What is the name of the longest river?'))
# print(convo.invoke('Who was the captain of the winning team?'))
# print(convo.memory.buffer)


memory_limited = ConversationBufferWindowMemory(k=1)
convo_limited = ConversationChain(
    llm=OpenAI(temperature=0),
    memory = memory_limited 
    )
# print(convo_limited.invoke('Who won the first cricket world cup?'))
# print(convo_limited.invoke('What is the name of the longest river?'))
# print(convo_limited.invoke('Who was the captain of the winning team?'))
# print(convo_limited.memory.buffer)

