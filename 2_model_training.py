import openai
import streamlit as strr

strr.sidebar.subheader("🤖 Career & Salary Chatbot (Powered by GPT)")

# 🔒 Prompt user to enter their OpenAI API key
api_key = strr.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# 📝 Question input
user_question = strr.sidebar.text_input("Ask your question:")

if user_question and api_key:
    with strr.spinner("🤖 Thinking..."):
        try:
            # Set API key dynamically
            openai.api_key = api_key
            
            # Make the API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert career advisor specialized in IT salaries, career paths, and negotiation strategies in India. Keep answers short, helpful, and to the point."},
                    {"role": "user", "content": user_question}
                ]
            )
            
            answer = response['choices'][0]['message']['content']
            strr.sidebar.markdown(f"💬 **Answer:** {answer}")

        except Exception as e:
            strr.sidebar.error(f"Error: {e}")

elif user_question and not api_key:
    strr.sidebar.warning("⚠️ Please enter your OpenAI API key to use the chatbot.")
