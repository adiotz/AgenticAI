from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
# 1. Initialize Grok (using the OpenAI-compatible interface)
llm = ChatOpenAI(
    model="grok-2-latest", 
    api_key=os.getenv("lab4_key"),
    base_url="https://api.x.ai/v1"  # This redirects requests to Grok
)

# --- STEP 1: Summarization Chain ---
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize this text into 3 vital bullet points for a busy executive:\n\n{text}"
)
summarize_chain = summarize_prompt | llm | StrOutputParser()

# --- STEP 2: Email Drafting Chain ---
email_prompt = ChatPromptTemplate.from_template(
    "Draft a professional email. Use the following summary as the body content. "
    "Recipient: {recipient}\nSummary:\n{summary}"
)
email_chain = email_prompt | llm | StrOutputParser()

# --- Automated Execution ---
def run_lab4_workflow(text_input, recipient):
    # Step 1: Summary
    print("Grok is summarizing...")
    summary = summarize_chain.invoke({"text": text_input})
    
    # Step 2: Email (Passing output of Step 1 to Step 2)
    print("Grok is drafting the email...")
    email = email_chain.invoke({
        "recipient": recipient,
        "summary": summary
    })
    
    return email

# Test Data
input_article = "The shift toward Agentic AI workflows allows for complex, autonomous task completion..."
print("\n--- Final Output ---")
print(run_lab4_workflow(input_article, "Aditya"))