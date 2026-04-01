import streamlit as st
import requests
from google import genai
from google.genai import types

# --- 1. Define the Tool (Agent Action) ---
def get_weather(city: str):
    """Retrieves the current weather for a given city."""
    try:
        # Geocoding: City Name -> Lat/Lon
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get('results'):
            return {"error": f"Could not find coordinates for {city}."}
        
        loc = geo_res['results'][0]
        lat, lon = loc['latitude'], loc['longitude']
        
        # Weather Fetch
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_res = requests.get(w_url).json()
        current = w_res['current_weather']
        
        return {
            "city": loc['name'],
            "country": loc.get('country', 'Unknown'),
            "temp": f"{current['temperature']}°C",
            "wind": f"{current['windspeed']} km/h"
        }
    except Exception as e:
        return {"error": str(e)}

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Weather Agent", page_icon="🌤️")
st.title("🌤️ Real-Time Weather Agent")
st.caption("Agentic AI Lab 2: Function Calling & Real-Time Interaction")

# Sidebar for API Key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.session_state.api_key = st.text_input("Gemini API Key", type="password")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 3. Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the weather in any city..."):
    if not st.session_state.api_key:
        st.error("Please enter your API Key in the sidebar!")
    else:
        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Agent Response Logic
        with st.chat_message("assistant"):
            with st.spinner("Agent is consulting tools..."):
                try:
                    client = genai.Client(api_key=st.session_state.api_key)
                    # Create chat with automatic tool use
                    chat = client.chats.create(
                        model="gemini-flash-latest",
                        config=types.GenerateContentConfig(tools=[get_weather])
                    )
                    
                    response = chat.send_message(prompt)
                    full_response = response.text
                    st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error: {e}")