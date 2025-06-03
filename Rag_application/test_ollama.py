
import requests
# Test the Ollama API
response = requests.post(
"http://localhost:11434/api/generate",
json={
"model": "llama3.2:1b",
"prompt": "Hello, what are your capabilities?",
"stream": False
}
)
print(response.json()["response"])
