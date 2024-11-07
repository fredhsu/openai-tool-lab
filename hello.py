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


def execute_tool_call(client: OpenAI, tool_call, messages: List[ChatCompletionMessageParam], tools: List[ChatCompletionToolParam]) -> str:
    """
    Execute a tool call and return the result.
    
    Args:
        client (OpenAI): The OpenAI client
        tool_call: The tool call to execute
        messages: The current conversation messages
        tools: Available tools
    
    Returns:
        str: The result of the tool call
    """
    arguments = json.loads(tool_call.function.arguments)
    function_name = tool_call.function.name
    
    print(f"{function_name} called with arguments", arguments)
    
    if function_name == "create_file":
        result = create_file(arguments["filename"], arguments["text"])
    elif function_name == "create_directory":
        result = create_directory(arguments["dirname"])
    elif function_name == "directory_exists":
        result = directory_exists(arguments["dirname"])
        
        # If directory_exists is called, get the next step from LLM
        messages.append({"role": "system", "content": result})
        next_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        
        print("Next response:", next_response)
        
        # Process the next response's tool calls if any
        if next_response.choices[0].message.tool_calls is not None:
            return execute_tool_call(client, next_response.choices[0].message.tool_calls[0], messages, tools)
    
    print(result)
    return result


def main():
    client = OpenAI()
    create_file_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a text file with a given filename and with the text provided. Use this function whenever the user asks to create a file with generated text",
            "strict": True,
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
                "additionalProperties": False,
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
        execute_tool_call(client, response.choices[0].message.tool_calls[0], messages, tools)


if __name__ == "__main__":
    main()
