import warnings
warnings.filterwarnings("ignore")
import os
import sys
import textwrap
import time
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import torch
from langchain_community.vectorstores import FAISS
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# 加入從命令行參數讀取文件路徑的代碼
question_path = sys.argv[1]
pdf_paths = sys.argv[2:]

class CFG:
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15
    do_sample = True
    max_new_tokens = 400
    num_return_sequences=1

    split_chunk_size = 800
    split_overlap = 0
    
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'
    k = 3
    PDFs_paths = pdf_paths
    Embeddings_path = './faiss_index_py'

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 初始化语言模型
llm = HuggingFaceEndpoint(
    repo_id = CFG.model_name,
    max_new_tokens = CFG.max_new_tokens,
    temperature = CFG.temperature,
    top_p = CFG.top_p,
    repetition_penalty = CFG.repetition_penalty,
    do_sample = CFG.do_sample,
    num_return_sequences = CFG.num_return_sequences
)

# 加载和分割PDF文件
documents = []
for pdf_path in CFG.PDFs_paths:
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents from {pdf_path}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CFG.split_chunk_size,
    chunk_overlap = CFG.split_overlap
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} text chunks")

# 创建 SentenceTransformer 模型
model = SentenceTransformer(CFG.embeddings_model_repo)
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': True}

# 创建嵌入
embeddings = HuggingFaceInstructEmbeddings(
    model_name=CFG.embeddings_model_repo,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = FAISS.from_documents(
    documents = texts, 
    embedding = embeddings
)
vectordb.save_local(CFG.Embeddings_path)
vectordb = FAISS.load_local(
    CFG.Embeddings_path,
    embeddings,
    allow_dangerous_deserialization=True
)

prompt_template = """
<s>[INST] 
Do not attempt to fabricate an answer. If the information is not available in the context, simply state that you don't know.
Answer in the same language the question was asked.
Provide a concise and accurate answer based strictly on the provided context.
Reference specific chemical theories, formulas, or data directly from the context.
Use technical and professional language suitable for a chemistry research paper.
Ensure the answer is precise and clearly references the provided context.
Quote specific sentences or data points from the context where applicable.
If the question cannot be answered based on the provided context, state explicitly that the information is not available in the provided context.

{context}

Question: {question}
Answer:[/INST]"""

PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["question", "context"]
)
llm_chain = LLMChain(prompt=PROMPT, llm=llm)
retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)

def wrap_text_preserve_newlines(text, width=700):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    print("LLM Response: ", llm_response)
    ans = wrap_text_preserve_newlines(llm_response['result'])
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    ans = ans + '\n\nSources: \n' + sources_used
    print("=================================")
    print(sources_used)
    return ans

def llm_ans(query):
    start = time.time()
    llm_response = qa_chain(query)
    if not llm_response:
        print("No response from llm_ans.")
        return ""
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans.strip() + time_elapsed_str

def extract_text_after_inst(input_string):
    marker_index = input_string.find("[/INST]")
    if marker_index != -1:
        extracted_text = input_string[marker_index + len("[/INST]"):].strip()
        print("Extracted text: ", extracted_text)  # 添加此行
        return extracted_text
    else:
        return ""
    

data = pd.read_excel(question_path)

if 'answer' not in data.columns:
    data['answer'] = ''

def predict(message, data_content=None):
    if data_content:
        message += f"\nData: {data_content}"
    output = str(llm_ans(message))
    #output = extract_text_after_inst(output)
    return output if isinstance(output, str) else ""

for index, row in data.iterrows():
    question = row['question']
    data_content = row.get('data', None)
    print(f"Processing question: {question}")
    answer = predict(question, data_content)
    if not answer:
        print(f"No answer generated for question: {question}")
    else:
        print(f"Generated answer: {answer}")
    data.at[index, 'answer'] = str(answer) if answer else ""

data.to_excel("answers.xlsx", index=False)

# 输出读取到的Excel文件的内容
print(data.head())
