import json
import os
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


def create_directory(dirname: str) -> str:
    """
    Create a directory with the given name.

    Args:
        dirname (str): Name of the directory to create

    Returns:
        str: Confirmation message
    """
    os.makedirs(dirname, exist_ok=True)
    return f"directory '{dirname}' created"


def directory_exists(dirname: str) -> str:
    """
    Check if a directory exists and return a descriptive response.

    Args:
        dirname (str): Path to the directory to check

    Returns:
        str: A human-readable response about the directory's existence
    """
    if os.path.isdir(dirname):
        return f"The directory '{dirname}' exists."
    else:
        return f"The directory '{dirname}' does not exist."


def main():
    client = OpenAI()
    create_file_definition: ChatCompletionToolParam = {
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

    directory_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory with the given name",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "dirname": {
                        "type": "string",
                        "description": "The name of the directory to create",
                    },
                },
                "additionalProperties": False,
                "required": ["dirname"],
            },
        },
    }

    directory_exists_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "directory_exists",
            "description": "Check if a directory exists",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "dirname": {
                        "type": "string",
                        "description": "The path to the directory to check",
                    },
                },
                "additionalProperties": False,
                "required": ["dirname"],
            },
        },
    }

    tools: List[ChatCompletionToolParam] = [
        create_file_definition,
        directory_definition,
        directory_exists_definition,
    ]

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful computer network designer. Your job is to take the user input and create a series of files that can be used to automate the deployment of a network. When creating a new file it should be placed in the directory 'group_vars'. If the directory does not exist, create it before creating the file. Use the supplied tools to assist the user. Reason through the steps and tools you will use step by step.",
        },
        {
            "role": "user",
            "content": "Create a new file for a campus network called `CAMPUS.yml`",
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
        elif function_name == "create_directory":
            print("create_directory called with arguments", arguments)
            dirname = arguments["dirname"]
            result = create_directory(dirname)
            print(result)
        elif function_name == "directory_exists":
            print("directory_exists called with arguments", arguments)
            dirname = arguments["dirname"]
            result = directory_exists(dirname)
            print(f"Directory '{dirname}' exists: {result}")
            
            # Add the directory existence result to messages
            messages.append({
                "role": "system", 
                "content": result
            })
            
            # Get the next step from the LLM
            next_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            
            print("Next response:", next_response)
            
            # Process the next response's tool calls if any
            if next_response.choices[0].message.tool_calls is not None:
                next_tool_call = next_response.choices[0].message.tool_calls[0]
                next_arguments = json.loads(next_tool_call.function.arguments)
                next_function_name = next_tool_call.function.name
                
                if next_function_name == "create_directory":
                    print("create_directory called with arguments", next_arguments)
                    dirname = next_arguments["dirname"]
                    result = create_directory(dirname)
                    print(result)
                elif next_function_name == "create_file":
                    print("create_file called with arguments", next_arguments)
                    filename = next_arguments["filename"]
                    text = next_arguments["text"]
                    result = create_file(filename, text)
                    print(result)


if __name__ == "__main__":
    main()
