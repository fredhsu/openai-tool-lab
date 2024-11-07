import json
from openai import BaseModel, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from typing import List


class NetworkDesignInput(BaseModel):
    network_type: str
    leafs: int
    spines: int


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
            "strict": True,  # used for structured outputs
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
                "additionalProperties": False,  # doesn't allow creation of key that isn't in the list
                "required": ["filename", "text"],
            },
        },
    }

    tools: List[ChatCompletionToolParam] = [function_definition]

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful computer network designer. Your job is to take the user input and create a series of files that can be used to automate the deployment of a network. Use the supplied tools to assist the user. ",
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
