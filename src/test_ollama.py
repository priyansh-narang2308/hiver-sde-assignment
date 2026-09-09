import ollama

def test_ollama_connection(model_name="gemma4:e2b"):
    try:
        print(f"Attempting to connect to Ollama using model: {model_name}...")
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': 'Say "Hello, world!" if you can read this.',
            },
        ])
        print("Success! Ollama responded:")
        print(response['message']['content'])
    except Exception as e:
        print("Failed to connect or generate response:")
        print(e)

if __name__ == "__main__":
    # Test with the specific model provided by user
    test_ollama_connection()
