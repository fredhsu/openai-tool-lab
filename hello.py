import json
import yaml
import os
from openai import BaseModel, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from typing import List


class NetworkDesignInput(BaseModel):
    network_type: str
    leafs: int
    spines: int


def create_inventory(network_design_input: NetworkDesignInput) -> str:
    spines = {}
    leafs = {}
    for i in range(network_design_input.spines):
        spines[f"SPINE{i}"] = None

    for i in range(network_design_input.leafs):
        leafs[f"LEAF{i}"] = None

    inventory = {
        "CAMPUS": {"children": {"SPINES": {"hosts": spines}, "LEAFS": {"hosts": leafs}}}
    }
    inventory_yaml = yaml.dump(inventory)

    return inventory_yaml.replace("null", "")


def add_service():
    pass


def get_services():
    pass


def validate_inputs():
    pass


def generate_avd():
    pass


def create_base_files(network_design_input: NetworkDesignInput):
    # Create inventory
    inventory = create_inventory(network_design_input)
    create_file("inventory.yaml", inventory)
    print("created inventory")
    # Create group_vars
    create_directory("group_vars")
    # TODO: Add make the choices between network types an enum and create the appropriate types in the files
    create_file("group_vars/SPINES.yaml", "type: l3spine")
    create_file("group_vars/LEAFS.yaml", "type: l2leaf")
    # Create campus - will collapse the variables for campus and fabirc to just campus
    campus_variables = """
    local_users:
      - name: admin
        privilege: 15
        role: network-admin
        sha512_password: "$6$eucN5ngreuExDgwS$xnD7T8jO..GBDX0DUlp.hn.W7yW94xTjSanqgaQGBzPIhDAsyAl9N4oScHvOMvf07uVBFI4mKMxwdVEUVKgY/."

    # AAA Authorization
    aaa_authorization:
      exec:
        default: local

    # OOB Management network default gateway.
    mgmt_gateway: 172.16.100.1
    mgmt_interface: Management0

    # Fabric settings
    fabric_name: CAMPUS

    # Spine Switches
    l3spine:
        defaults:
            platform: cEOSLab
            loopback_ipv4_pool: 172.16.1.0/24
        node_groups:
            - group: SPINES
            nodes:
                - name: SPINE1
                id: 1
                mgmt_ip: 172.16.100.101/24
                - name: SPINE2
                id: 2
                mgmt_ip: 172.16.100.102/24

    # IDF - Leaf Switches
    l2leaf:
        defaults:
            platform: cEOSLab
            mlag_peer_ipv4_pool: 192.168.0.0/24
            spanning_tree_mode: mstp
            spanning_tree_priority: 16384
            inband_mgmt_subnet: 10.10.10.0/24
            inband_mgmt_vlan: 10
        node_groups:
            - group: IDF1
            mlag: true
            uplink_interfaces: [Ethernet51]
            mlag_interfaces: [Ethernet53, Ethernet54]
            filter:
                tags: [ "110", "120", "130" ]
            nodes:
                - name: LEAF1A
                id: 3
                mgmt_ip: 172.16.100.103/24
                uplink_switches: [SPINE1]
                uplink_switch_interfaces: [Ethernet1]
                - name: LEAF1B
                id: 4
                mgmt_ip: 172.16.100.104/24
                uplink_switches: [SPINE2]
                uplink_switch_interfaces: [Ethernet1]
    """

    create_file("group_vars/CAMPUS.yaml", campus_variables)


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


def execute_tool_call(
    client: OpenAI,
    tool_call,
    messages: List[ChatCompletionMessageParam],
    tools: List[ChatCompletionToolParam],
) -> str:
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
            return execute_tool_call(
                client, next_response.choices[0].message.tool_calls[0], messages, tools
            )
    else:
        result = "No function with that name was found"

    print(result)
    return result


def main():
    client = OpenAI()

    network_design_input_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "network_design_input",
            "description": "Creates a series of files to assist with deploying a network using Ansible.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "network_type": {
                        "type": "string",
                        "description": "The type of network design, should be one of layer 3, layer 2, or campus",
                    },
                    "leafs": {
                        "type": "int",
                        "description": "The number of leaf switches in the network",
                    },
                    "spines": {
                        "type": "int",
                        "description": "the number of spine switches in the network",
                    },
                },
                "additionalProperties": False,
                "required": ["network_type", "leafs", "spines"],
            },
        },
    }

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
            "content": "You are a helpful computer network designer. Your job is to take the user input and create a series of files that can be used to automate the deployment of a network. When creating a new file it should be placed in the directory 'group_vars'. If the directory does not exist, create it before creating the file. Use the supplied tools to assist the user. Reason through the steps and tools you will use step by step. If the user does not provide the necessary information, prompt them for the missing information",
        },
        {
            "role": "user",
            "content": "Create a new file for a campus network with 8 leafs and 2 spines",
        },
    ]
    # messages: List[ChatCompletionMessageParam] = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful computer network designer. Your job is to take the user input and create a series of files that can be used to automate the deployment of a network. When creating a new file it should be placed in the directory 'group_vars'. If the directory does not exist, create it before creating the file. Use the supplied tools to assist the user. Reason through the steps and tools you will use step by step. If the user does not provide the necessary information, prompt them for the missing information",
    #     },
    #     {
    #         "role": "user",
    #         "content": "Create a new file for a campus network called `CAMPUS.yaml`",
    #     },
    # ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="required",
    )
    print(response)

    if response.choices[0].message.tool_calls is None:
        print("No tool calls found")
    else:
        execute_tool_call(
            client, response.choices[0].message.tool_calls[0], messages, tools
        )


if __name__ == "__main__":
    main()
