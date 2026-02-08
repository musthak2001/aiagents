import json
import os
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional

# -------------------------------
# Load API key
# -------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# Load memory (if exists)
# -------------------------------
MEMORY_FILE = "chat_memory.json"

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        messages = json.load(f)
else:
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant you can answer general questions also"}
    ]

# -------------------------------
# Define the tool (function)
# -------------------------------
def get_weather(latitude, longitude):
    """Returns current weather for given location"""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

# -------------------------------
# Define tools
# -------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

# -------------------------------
# User input
# -------------------------------
user_input = input("Ask the weather assistant: ")
messages.append({"role": "user", "content": user_input})

# -------------------------------
# Call AI model
# -------------------------------
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# -------------------------------
# Function to execute tool calls
# -------------------------------
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

# -------------------------------
# Execute tool calls
# -------------------------------
tool_calls = getattr(completion.choices[0].message, "tool_calls", None)

if tool_calls:
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # Run the tool
        result = call_function(name, args)

        # Add tool output to memory
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )

        # Add AI response again after tool execution
        messages.append(completion.choices[0].message.model_dump())
else:
    # No tool call → just store AI message
    messages.append(completion.choices[0].message.model_dump())

# -------------------------------
# Parse AI response
# -------------------------------
class AIResponse(BaseModel):
    temperature: Optional[float] = Field(description="Current temperature in Celsius, if applicable")
    response: str = Field(description="Natural language response from AI")

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=AIResponse,
)

final_response = completion_2.choices[0].message.parsed

# Print results
if final_response.temperature is not None:
    print("Temperature:", final_response.temperature)
print("AI says:", final_response.response)
# -------------------------------
# Save memory for next session
# -------------------------------

with open(MEMORY_FILE, "w") as f:
    json.dump(messages, f, indent=2, separators=(',', ': '))

