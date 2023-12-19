from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
import os


"""_Libraries_
1. Make sure you have cuda and cudnn enabled system if not then follow this as per your OS: https://techzizou.com/install-cuda-and-cudnn-on-windows-and-linux/
2. Install Pytorch into your system: https://pytorch.org/       # make sure to select nightly version 12.x
3. Then Run these commands in your terminal 1 by 1:
    i). pip3 install transformers
    ii). pip3 install huggingface-hub
    iii). pip3 install langchain[all] duckduckgo-search
    iv). pip3 install openai
    v). pip3 install --upgrade transformers optimum
    vi). pip3 uninstall -y auto-gptq
    vii). mkdir LLMModels
    viii). cd LLMModels
    ix). git clone https://github.com/PanQiWei/AutoGPTQ
    x). cd AutoGPTQ
    xi). git checkout v0.5.1
    xii). pip3 install gekko
    xiii). pip3 install .
"""


model_name_or_path = "LLMModels/Mistral-7B-Instruct-v0.2-DARE-GPTQ"

# Check if the model weights exists
if not os.path.exists(model_name_or_path):
    # If not, create the directory
    os.makedirs(model_name_or_path)
    print(f"The directory '{model_name_or_path}' has been created.\n\n")


    # To use a different branch, change revision
    # For example: revision="gptq-8bit-32g-actorder_True" See this page for further understanding: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GPTQ

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-DARE-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="gptq-8bit-32g-actorder_True")
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,                                                   ############# make it comment if you don't want stream the output
               )
    pipe.save_pretrained(model_name_or_path)

else:
    print(f"The directory '{model_name_or_path}' already exists.\n\n")

    # To use a different branch, change revision
    # For example: revision="gptq-8bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="gptq-8bit-32g-actorder_True")
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,                                        ###################    make it comment if you don't want stream the output
               )           

llm = HuggingFacePipeline(pipeline=pipe)

search = DuckDuckGoSearchRun()

diseases = ["Pneumonia"]  # this will be your CNN model's output class/classes stored in the list i am pasting here for just a demo


def generate_with_Rag(user_input):
    
    context = search.run(user_input)

    system_message = """You are a helpful, respectful and honest assistant. Answer the question in short and precise manner with best of your ability and use context if question is related to context else plain answer without using context."""
    template = """system
    {system_message}
    context
    {context}
    question
    {question}
    assistant
    """

    question = f"""{user_input}"""
    prompt = PromptTemplate(template=template, input_variables=["system_message", "question", "context"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"system_message":system_message, "context":context, "question":question})
    # Remove </s> from the response
    response = response.replace('</s>', '')
    return response


def take_class(_class):
    q1 = f"What is {_class}"
    q2 = f"What are symptoms of {_class}"
    q3 = f"What are causes of {_class}"
    q4 = f"What is the treatment for {_class}"
    return q1, q2, q3, q4

if __name__ == "__main__":

    for i in diseases:
        if i == "No findings":
            pass
        else:
            q1, q2, q3, q4 = take_class(i)
            print(f"Question1: {q1}")
            print(generate_with_Rag(q1))

            print(f"\nQuestion2: {q2}")
            print(generate_with_Rag(q2))

            print(f"\nQuestion3: {q3}")
            print(generate_with_Rag(q3))

            print(f"\nQuestion4: {q4}")
            print(generate_with_Rag(q4)) 