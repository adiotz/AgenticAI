import requests
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
# 1. Define the "Tool" (The actual Python function)
def get_weather(city: str):
    """Retrieves the current weather for a given city."""
    # First, get coordinates for the city (Geocoding)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    geo_res = requests.get(geo_url).json()
    
    if not geo_res.get('results'):
        return f"Could not find coordinates for {city}."
    
    lat = geo_res['results'][0]['latitude']
    lon = geo_res['results'][0]['longitude']
    
    # Second, get the weather for those coordinates
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_res = requests.get(weather_url).json()
    
    current = weather_res['current_weather']
    return {
        "city": city,
        "temperature": current['temperature'],
        "windspeed": current['windspeed'],
        "condition_code": current['weathercode']
    }

# 2. Initialize Gemini Client with the tool
client = genai.Client(api_key= os.getenv("lab2_key"))

# Create a chat session with 'Automatic Function Calling' enabled
# This allows Gemini to run the get_weather function and see the result automatically
chat = client.chats.create(
    model="gemini-flash-latest",
    config=types.GenerateContentConfig(
        tools=[get_weather] # Register your function here
    )
)

# 3. Working Chatbot Loop
print("--- Weather Agent Ready (Type 'exit' to quit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
        
    response = chat.send_message(user_input)
    print(f"Agent: {response.text}")