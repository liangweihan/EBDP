import subprocess
import sys
import PyPDF2
import google.generativeai as genai
import re
import os
import pandas as pd

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing requirements: {e}")
        sys.exit(1)

api_key = 'AIzaSyBwV30tJyjUL1Zk7xMA9MILOUhIDvlbpvk'
genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_file, page_numbers):
    text = ""
    if not os.path.exists(pdf_file):
        print(f"File not found: {pdf_file}")
        return text

    try:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            for page_num in page_numbers:
                if page_num < num_pages:
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
                else:
                    print(f"Page number {page_num + 1} is out of range for file {pdf_file}")
    except Exception as e:
        print(f"An error occurred while reading {pdf_file}: {e}")
    return text

def chatting(text, instruction):
    LAB_prompt = text + " This is my chemistry paper and data, give me a simple analysis and answer"
    final_prompt = LAB_prompt + instruction
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(final_prompt)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"An error occurred while generating content: {e}")
        return ""

def parse_input(input_str):
    match = re.match(r"(.+?) - page: ([\d, ]+)", input_str)
    if match:
        pdf_file = match.group(1).strip() + ".pdf"
        page_numbers = [int(x.strip()) - 1 for x in match.group(2).split(",")]
        return pdf_file, page_numbers
    return "", []

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        questions = df['question'].tolist()
        sources = df['answer'].tolist()
        return questions, sources
    except Exception as e:
        print(f"An error occurred while reading Excel file: {e}")
        return [], []

def extract_sources(answer):
    sources = re.findall(r"Sources:\s*(.+)", answer, re.MULTILINE | re.DOTALL)
    if sources:
        return sources[0].strip().split('\n')
    return []

def append_data_if_exists(question, data_content):
    if pd.notna(data_content):
        return f"{question}\nData: {data_content}"
    return question

excel_file = r"answers.xlsx"
questions, answers = read_excel(excel_file)
responses = []

data = pd.read_excel(excel_file)

for index, row in data.iterrows():
    question = row['question']
    data_content = row.get('data', None)
    question = append_data_if_exists(question, data_content)
    
    sources = extract_sources(row['answer'])
    all_text = ""
    for source in sources:
        pdf_file, page_numbers = parse_input(source.strip())
        if pdf_file and page_numbers:
            text = extract_text_from_pdf(pdf_file, page_numbers)
            all_text += text + "\n\n"
    
    if not all_text.strip():
        print(f"No text extracted from the provided sources for question: {question}")
        continue

    generated_answer = chatting(all_text, question)
    response_text = f"Question: {question}\nAnswer: {generated_answer}\nSources used:\n{row['answer']}\n"
    responses.append(response_text)
    print(response_text)
    print("======================================================")

with open("gemini_responses.txt", "w", encoding="utf-8") as f:
    for response in responses:
        f.write(response)
print("已完成所有問題的回答")
