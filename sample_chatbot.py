import openai

# Load your OpenAI API key securely
api_key = "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"  # Replace with your actual API key

client = openai.OpenAI(api_key=api_key)

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3",  # or "gpt-3.5-turbo" for a cheaper option
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        bot_response = chat_with_gpt(user_input)
        print(f"Bot: {bot_response}")
