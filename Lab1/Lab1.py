from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key= os.getenv("lab1_key"))

def summarize_text(text):
    response = client.models.generate_content(
        model="gemini-flash-latest", 
        contents=f"Please provide a concise summary of this text:\n\n{text}"
    )
    return response.text

my_text = """
Agentic AI refers to artificial intelligence systems designed to act as autonomous agents. 
Unlike standard LLMs that just respond to prompts, Agentic AI can plan multi-step tasks, 
use tools (like web search or calculators), and iterate on its own work to achieve a 
specific goal defined by the user.
"""

print("--- Summary ---")
print(summarize_text(my_text))