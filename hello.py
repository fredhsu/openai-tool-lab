import openai
from openai import OpenAI

def create_file(filename: str, text: str):
    with open("test.txt", "w") as file:
        file.write("foo")

def main():
    client = OpenAI()
    function_definition = """
    {
        "name": "create_file",
        "description": "Create a text file with a given filename and with 
        the text provided. Use this function whenever the user asks to 
        create a file with generated text",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to use when creating a file"
                },

                "text": {
                    "type": "string",
                    "description": "The text to write to the file"
                }
            },
            "required": ["filename", "text"],
            "additionalProperties": false
        }
    }
        """
    tools = [function_definition ]

    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
        {"role": "user", "content": "Create a new file called `test.txt"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    print(response)


if __name__ == "__main__":
    main()
