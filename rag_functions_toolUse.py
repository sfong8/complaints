import boto3
import json
import math
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import pandas as pd

# Initialize the Bedrock runtime client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',
)

# Create an instance of BedrockEmbeddings using the Bedrock client
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

# Load the previously created FAISS vector store for complaints
vector_store = FAISS.load_local('complaints.vs', embeddings, allow_dangerous_deserialization=True)

def getContext(user_prompt: str, filter_terms: dict) -> pd.DataFrame:
    """
    Retrieves the context (complaints) based on the user's query.

    Parameters:
    user_prompt (str): The user's query.
    filter_terms (dict): Dictionary containing metadata filter and k_filter.

    Returns:
    pd.DataFrame: A DataFrame containing the retrieved complaints.
    """
    try:
        metadata_filter = filter_terms['metadata_filter']
        k_filter = filter_terms['k_filter']
        print(f"Filter by metadata: {metadata_filter} and k: {k_filter}")
        retriever = vector_store.as_retriever(search_kwargs={'filter': metadata_filter, 'k': k_filter})
    except Exception as e:
        print('Cannot extract filters, using default settings.')
        retriever = vector_store.as_retriever(search_kwargs={'k': 100})

    docs = retriever.invoke(user_prompt)
    master_df = pd.DataFrame()

    for doc in docs:
        test = doc.metadata
        test.update({'complaint_text': doc.page_content})
        temp_df = pd.DataFrame([test.values()], columns=test.keys())
        master_df = pd.concat([master_df, temp_df])

    return master_df

def getResponse(user_prompt: str, context_df: pd.DataFrame) -> str:
    """
    Generates a response to the user's query using the retrieved context.

    Parameters:
    user_prompt (str): The user's query.
    context_df (pd.DataFrame): The DataFrame containing the retrieved context.

    Returns:
    str: The generated response.
    """
    message_list = [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

    rag_system_message = f"""
    System: You are an AI assistant in a corporate bank. Your job is to answer the user's query around complaints using only the provided context. 
    Mainly use the complaint_text column to generate answers but can use other columns to check the user query. 
    Human: Here is a set of context, contained in <context> tags:
    
    <context>
    {context_df.to_csv(index=None)}
    </context>
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """

    response = bedrock_client.converse(
        modelId="amazon.nova-pro-v1:0",
        messages=message_list,
        system=[{'text': rag_system_message}],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.1
        }
    )

    return response['output']['message']['content'][0]['text']

def call_bedrock(message_list, system_prompts, tool_list) -> dict:
    """
    Interacts with Amazon Bedrock Converse API to generate responses using a specified model and tools.

    Parameters:
    message_list (list): A list of messages to send to the model.
    system_prompts (str): System prompts to guide the model's response.
    tool_list (list): A list of tools to be used by the model.

    Returns:
    dict: The response from the Bedrock service.
    """
    response = bedrock_client.converse(
        modelId="amazon.nova-pro-v1:0",
        messages=message_list,
        system=[{'text': system_prompts}],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.1
        },
        toolConfig={"tools": tool_list}
    )

    return response

# Define the tool list
tool_list = [
    {
        "toolSpec": {
            "name": "identify_complaints_filters",
            "description": """Identify filters for complaints based on the query.""",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "filter_dict": {
                            "type": "dict",
                            "description": """Output filter as a dict e.g. {'metadata_filter': {'client_region': ['MC', 'LC']}, 'k_filter': 5}"""
                        }
                    },
                    "required": ["filter_dict"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_complaintsData",
            "description": """Retrieve the context (complaints) based on the user's query.""",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "str",
                            "description": """Original user query"""
                        },
                        "filter_terms": {
                            "type": "dict",
                            "description": """Filter terms from the output of tool identify_complaints_filters"""
                        }
                    },
                    "required": ["user_query", "filter_terms"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "generateResponse",
            "description": """Generate a response based on the user's query and the retrieved context (complaints). Only use this if the user wants insights or summary of the complaints.""",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "str",
                            "description": """Original user query"""
                        },
                        "context_df": {
                            "type": "string",
                            "description": """Context DataFrame from the output of tool get_complaintsData"""
                        }
                    },
                    "required": ["user_query", "context_df"]
                }
            }
        }
    }
]

def handle_response(response_message):
    """
    Handles the response from the Bedrock service and prepares follow-up content blocks.

    Parameters:
    response_message (dict): The response message from the Bedrock service.

    Returns:
    dict: The follow-up message to be sent to the Bedrock service.
    """
    response_content_blocks = response_message['content']
    follow_up_content_blocks = []

    for content_block in response_content_blocks:
        if 'toolUse' in content_block:
            tool_use_block = content_block['toolUse']

            if tool_use_block['name'] == 'identify_complaints_filters':
                try:
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [
                                {"json": {"filter_terms": tool_use_block['input']}}
                            ]
                        }
                    })
                except Exception as e:
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [{"text": repr(e)}],
                            "status": "error"
                        }
                    })

            if tool_use_block['name'] == 'get_complaintsData':
                user_query = tool_use_block['input']['user_query']
                filter_terms = tool_use_block['input']['filter_terms']
                try:
                    context_df = getContext(user_query, filter_terms)
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [
                                {"json": {"context_df": context_df.to_csv(index=None)}}
                            ]
                        }
                    })
                except Exception as e:
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [{"text": repr(e)}],
                            "status": "error"
                        }
                    })

            if tool_use_block['name'] == 'generateResponse':
                user_query = tool_use_block['input']['user_query']
                context_df_str = tool_use_block['input']['context_df']
                try:
                    response = getResponse(user_query,context_df_str)
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [
                                {"text": response}
                            ]
                        }
                    })
                except Exception as e:
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [{"text": repr(e)}],
                            "status": "error"
                        }
                    })

    if follow_up_content_blocks:
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
        return follow_up_message
    else:
        return None

def run_loop(prompt, system_prompts):
    """
    Runs the main loop to interact with the Bedrock service.

    Parameters:
    prompt (str): The initial user prompt.
    system_prompts (str): System prompts to guide the model's response.
    tool_list (list): A list of tools to be used by the model.

    Returns:
    list: The final message list containing the conversation.
    """
    MAX_LOOPS = 10
    loop_count = 0
    continue_loop = True

    message_list = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    while continue_loop:
        response = call_bedrock(message_list, system_prompts, tool_list)
        response_message = response['output']['message']
        message_list.append(response_message)
        loop_count += 1

        if loop_count >= MAX_LOOPS:
            print(f"Hit loop limit: {loop_count}")
            break

        follow_up_message = handle_response(response_message)

        if follow_up_message is None:
            continue_loop = False
        else:
            message_list.append(follow_up_message)

    return message_list