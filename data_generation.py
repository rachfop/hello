import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/jhurblhokasqtv/openai/v1",
)
import random

def generate_example(prompt, prev_examples, temperature=0.9):
    messages = [
        {
            "role": "system",
            "content": f"""You are generating data to train a machine learning model.

            Given a high-level description of the desired model, generate diverse and high-quality prompt/response pairs.

            Format each example as follows:
            ```
            prompt
            -----------
            $prompt_goes_here
            -----------

            response
            -----------
            $response_goes_here
            -----------
            ```

            Generate one prompt/response pair per turn, gradually increasing complexity while maintaining diversity.

            Ensure the samples are unique, diverse, and sufficiently complex to train a well-performing model.

            Model description:
            `{prompt}`""",
        }
    ]

    if prev_examples:
        prev_examples = random.sample(prev_examples, min(len(prev_examples), 10))
        for example in prev_examples:
            messages.append({"role": "assistant", "content": example})

    response = client.chat.completions.create(
        model="openchat/openchat-3.5-1210",
        messages=messages,
        temperature=temperature,
        max_tokens=1354,
    )

    if response.choices and response.choices[0].message:
        return response.choices[0].message.content
    else:
        print("Error: Failed to generate example. Returning empty string.")
        return ""


def generate_system_message(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="openchat/openchat-3.5-1210",
        messages=[
            {
                "role": "system",
                "content": """Generate a concise system prompt for the model described below to use during inference.

                Use the format: Given $INPUT_DATA, you will respond in with A2 to B1 Brazilian Portuguese for ease of learning.

                Include only the system prompt in your response, without any additional text or formatting.

                Model description:
                """,
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message.content
