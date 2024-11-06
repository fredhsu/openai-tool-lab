import json
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from typing import List


def create_file(filename: str, text: str) -> str:
    with open(filename, "w") as file:
        file.write(text)
    return "file created"


def main():
    client = OpenAI()
    function_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a text file with a given filename and with the text provided. Use this function whenever the user asks to create a file with generated text",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename to use when creating a file",
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to write to the file",
                    },
                },
                "required": ["filename", "text"],
            },
        },
    }

    tools: List[ChatCompletionToolParam] = [function_definition]

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
        },
        {"role": "user", "content": "Create a new file called `test.txt`"},
        {
            "role": "user",
            "content": "Write a poem about transformers to be stored in the file",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    print(response)

    if response.choices[0].message.tool_calls is None:
        print("No tool calls found")
    else:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        function_name = tool_call.function.name
        if function_name == "create_file":
            print("create_file called with arguments", arguments)
            filename = arguments["filename"]
            text = arguments["text"]
            result = create_file(filename, text)
            print(result)


if __name__ == "__main__":
    main()
