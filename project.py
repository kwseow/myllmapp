from dotenv import load_dotenv
import os
from langchain.llms import Replicate
from PIL import Image 
import requests
from io import BytesIO
import secrets
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


load_dotenv('env03-2.txt')
openai_api_key = os.getenv('OPENAI_API_KEY')
replicate_app_id = os.getenv('REPLICATE_API_TOKEN')
# print the key
print("OpenAI Key:%s"%openai_api_key)
print("Replicate key:%s"%replicate_app_id)

dolly_llm = Replicate(
    model = "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    model_kwargs = {"temperature":0.80}
    )


t2i = Replicate(
    model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    )

first_prompt = PromptTemplate(
    input_variables=["company_name","product_name"],
    template="{company_name} is a company based in Singapore that sells {product_name}. Write a short 50 words advertisement for {company_name} to sell {product_name}.",
)

first_chain = LLMChain(
    llm = dolly_llm,
    prompt = first_prompt,
    output_key = "advert_text"
)

second_prompt = PromptTemplate(
   input_variables=["advert_text"],
   template="Generate a company logo based on the provided text: {advert_text}" 
) 

second_chain = LLMChain(
    llm = t2i,
    prompt = second_prompt,
    output_key = "logo_url"
)

overall_chain = SequentialChain(
    chains = [first_chain,second_chain],
    input_variables = ["company_name","product_name"],
    output_variables= ["logo_url","advert_text"],
    verbose=True
)

sequential_chain_response = overall_chain(
    {"company_name" : "Dolly Inc","product_name":"Fashion designer clothing"}
)

advert_text = sequential_chain_response["advert_text"]
logo_url = sequential_chain_response["logo_url"]
print("[info]%s"%advert_text)
print("[info]%s"%logo_url)

# download the image
response = requests.get(logo_url)
img = Image.open(BytesIO(response.content))

# save the image
filename = "image_%s.png"%secrets.token_hex(4)
img.save("./genImages/%s"%filename)
print("Image saved to %s"%filename)

