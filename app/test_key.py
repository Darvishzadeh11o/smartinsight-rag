from openai import OpenAI
from config import OPENAI_API_KEY

# Create OpenAI client using the key from config.py
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}]
)

print(response.choices[0].message.content)
