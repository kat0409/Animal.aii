import numpy as np
import pandas as pd  
import os
from transformers import pipeline
import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent='HackTX-AnimalAI (kathiana119@gmail.com)', language='en')
nlp = pipeline('question-answering', model='deepset/roberta-base-squad2')

def extract_relevant_text(page, question):
    keywords = ['physical characteristics', 'description', 'diet', 'mannerisms', 'appearance', 'eat', 'lifespan']
    question_keywords = set(question.lower().split())

    relevant_text = ""

    for section in page.sections:
        if any(keyword in section.title.lower() for keyword in keywords) or \
           any(keyword in section.title.lower() for keyword in question_keywords):
            relevant_text += section.text

    if not relevant_text:
        relevant_text = page.summary

    return relevant_text

def answer_question(topic, question):
    page = wiki.page(topic)

    if not page.exists():
        return "Page not found for the given topic."
    
    wiki_text = extract_relevant_text(page, question)

    max_token_length = 512
    if len(wiki_text.split()) > max_token_length:
        wiki_text = ' '.join(wiki_text.split()[:max_token_length])

    qa_input = {
        'question': question,
        'context': wiki_text
    }

    try:
        answer = nlp(qa_input)
        return answer['answer']
    except Exception as e:
        return f"An error occurred while answering the question: {str(e)}"

topic = input("Enter the topic you want information on: ")
question = input("Enter your question: ")

response = answer_question(topic, question)
print("Answer:", response)
