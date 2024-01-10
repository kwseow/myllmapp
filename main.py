from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI 
from lunary import LunaryCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# load_dotenv('env03-2.txt')
# openai_api_key = os.getenv('OPENAI_API_KEY')
lunary_app_id = os.getenv('LUNARY_APP_ID')
# # print the key
# print("OpenAI Key:%s"%openai_api_key)
print("Lunary key:%s"%lunary_app_id)

# set up lunary callback
handler = LunaryCallbackHandler(app_id=lunary_app_id)


# meal template
meal_template = PromptTemplate(
    input_variables=["ingredients"],
    template="Give me an example of 2 meals that could be made using the following ingredients: {ingredients}",
)

# gangster template
gangster_template = """Re-write the meals given below in the style of a New York mafia gangster:

Meals:
{meals}
"""

gangster_template = PromptTemplate(
    input_variables=['meals'],
    template=gangster_template
)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.9, callbacks=[handler])

meal_chain = LLMChain(
    llm=llm,
    prompt=meal_template,
    output_key="meals",  # the output from this chain will be called 'meals'
    verbose=True
)

gangster_chain = LLMChain(
    llm=llm,
    prompt=gangster_template,
    output_key="gangster_meals",  # the output from this chain will be called 'gangster_meals'
    verbose=True
)

overall_chain = SequentialChain(
    chains=[meal_chain, gangster_chain],
    input_variables=["ingredients"],
    output_variables=["meals", "gangster_meals"],
    verbose=True
)

st.title("Meal planner")

ingredients = st.text_input("Ingredients", "eggs, bacon, bread, cheese")

if st.button("Plan meals") and ingredients:
    with st.spinner("Generating...."):
        output = overall_chain({'ingredients': ingredients})

        # version 1
        # st.write(output["gangster_meals"])

        # version 2
        col1, col2 = st.columns(2)
        col1.write(output["meals"])
        col2.write(output["gangster_meals"])

