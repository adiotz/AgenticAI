import streamlit as st
from google import genai

# Page Config
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")

# --- UI Sidebar ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# --- UI Main Body ---
st.title("📝 Agentic AI: Text Summarizer")
st.markdown("Paste a long article below and let Gemini condense it for you.")

input_text = st.text_area("Input Text", placeholder="Paste your text here...", height=300)

if st.button("Generate Summary"):
    if not api_key:
        st.error("Please provide an API key in the sidebar!")
    elif not input_text:
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Gemini is thinking..."):
            try:
                # Initialize Client
                client = genai.Client(api_key=api_key)
                
                # Using 2.0-flash if available, else 1.5-flash
                # Note: 'gemini-1.5-flash' is the safest for free-tier in 2026
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=f"Summarize the following text in bullet points:\n\n{input_text}"
                )
                
                # Display Result
                st.subheader("Summary Result")
                st.success("Success!")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.divider()
st.caption("Developed for Agentic AI Lab - Semester 6")