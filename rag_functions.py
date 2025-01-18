import boto3, json, math
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

# Define the tool list for identifying complaints filters
# This tool is used to determine if the user's query requires filtering the complaints based on metadata
tool_list =[
    {
        "toolSpec": {
            "name": "identify_complaints_filters",
            "description": """Your job is to first look at the query's query and determine if it requires filtering the complaints first based off the following columns 
            client_name - name of the company making the complaint, you will need to use the like filter since users might not type if the exact name
            client_region - possible values ['MC','LC','ICB'] where MC= mid-corporate, LC = Large corporate, ICB = International corporate
            complaint_date - date of the complaint

            Note: if you're not sure then don't output anything. This is used for metadata filter for the vectorstore. There are two outputs x and y, where x is the column filters and y is the number of documents to return - if user doesnt specify then default is 100

            Example1: user_query: Show me all complaints in region MC
            output: = {client_region": "MC"}
            y=100

            Example2: user_query: Show me all complaints in region LC and after 15th June 2024
            output: 
            x = {"client_region": "MC","complaint_date": ">2024-06-15"}
            y=100

            Example3: user_query: Show me 5 complaints in regions MC and LC
            output:
            x = {client_region": ["MC",'LC']}
            y = 5
            
            """,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "dict",
                            "description": """output filter as a dict e.g. {client_region": ["MC",'LC']}  or {client_region": "MC"}"""
                        },
                        "y": {
                            "type": "integer",
                            "description": """Filter for the number of documents to return. if user query doesnt specify then default is 100"""
                        }
                    },
                    "required": ["x","y"]
                }
            }
        }
    }
]

def call_bedrock(message_list, system_prompts, extract_filter=True):
    """
    This function interacts with Amazon Bedrock service to generate responses using a specified model.

    Parameters:
    message_list (list): A list of messages to send to the model.
    system_prompts (str): System prompts to guide the model's response.
    tool_list (list): A list of tools to be used by the model (if applicable).
    extract_filter (bool, optional): Flag to determine whether to use tool configuration. Defaults to True.

    Returns:
    dict: The response from the Bedrock service.
    """
    
    # Create a boto3 session
    session = boto3.Session()

    # Initialize the Bedrock runtime client
    bedrock = session.client(service_name='bedrock-runtime')
    
    # Check if extract_filter is True to decide whether to include tool configuration
    if extract_filter:
        # Call the Bedrock converse API with tool configuration
        response = bedrock.converse(
            modelId="amazon.nova-pro-v1:0",  # ID of the model to use
            messages=message_list,  # Messages to send to the model
            system=[{ 'text': system_prompts }],  # System prompts
            inferenceConfig={  # Inference configuration
                "maxTokens": 2000,  # Maximum number of tokens to generate
                "temperature": 0.1  # Temperature for response randomness
            },
            toolConfig={ "tools": tool_list }  # Tool configuration
        )
    else:
        # Call the Bedrock converse API without tool configuration
        response = bedrock.converse(
            modelId="amazon.nova-pro-v1:0",  # ID of the model to use
            messages=message_list,  # Messages to send to the model
            system=[{ 'text': system_prompts }],  # System prompts
            inferenceConfig={  # Inference configuration
                "maxTokens": 2000,  # Maximum number of tokens to generate
                "temperature": 0.1  # Temperature for response randomness
            }
        )
    
    # Return the response from the Bedrock service
    return response

def getContext(user_prompt:str):
    """
    This function retrieves the context (complaints) based on the user's query.

    Parameters:
    user_prompt (str): The user's query.
    vector_store: The FAISS vector store containing the complaints.
    tool_list (list): The list of tools to use for filtering.

    Returns:
    pd.DataFrame: A DataFrame containing the retrieved complaints.
    """
    
    system_message_filter = """You are an AI assistant within a corporate bank in the Complaints team. Your role is to retrieve back the complaints based off the user query. 
    Your first job is always to breakdown the user query (using the tool identify_complaints_filters) """
    
    message_list = [
            {
                "role": "user",
                "content": [ { "text": user_prompt } ]
            }
        ]

    # Call the Bedrock service to get the filter response
    filter_response = call_bedrock(message_list=message_list,system_prompts=system_message_filter,extract_filter=True)
    print(filter_response)
    
    try:
        # Extract the metadata filter and the number of documents to return from the filter response
        metadata_filter = filter_response['output']['message']['content'][1]['toolUse']['input']['x']
        k_filter = filter_response['output']['message']['content'][1]['toolUse']['input']['y']
        print(f"filter by metadata: {metadata_filter} and k: {k_filter}")
        
        # Create a retriever with the extracted filters
        retriever = vector_store.as_retriever(search_kwargs={'filter': metadata_filter, 'k':k_filter})
    except:
        print('cannot extract out filters so going default')
        # Create a retriever with default settings if filters cannot be extracted
        retriever = vector_store.as_retriever(search_kwargs= {'k':100})

    # Retrieve the documents using the retriever
    docs = retriever.invoke(user_prompt)
    
    # Create a DataFrame to store the retrieved complaints
    master_df = pd.DataFrame()
    for doc in docs:
        test=doc.metadata
        test.update({'complaint_text':doc.page_content})
        temp_df = pd.DataFrame([test.values()],columns=test.keys())
        master_df = pd.concat([master_df,temp_df])
    
    return master_df

def getResponse(user_prompt:str,context_df):
    """
    This function generates a response to the user's query using the retrieved context.

    Parameters:
    user_prompt (str): The user's query.
    context_df (pd.DataFrame): The DataFrame containing the retrieved context.

    Returns:
    str: The generated response.
    """
    
    message_list = [
            {
                "role": "user",
                "content": [ { "text": user_prompt } ]
            }
        ]
    
    # Create a system message with the context
    rag_system_message = f"""
    System: You are an AI assistant in a corporate bank and your job is to answer the users query around complaints using only the context only you should mainly use the complaint_text column to generate answer but can use other columns to check the user query. 
    Human: Here is a set of context, contained in <context> tags:
    
    <context>
    {context_df.to_csv(index=False)}
    </context>
    
     If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """
    
    # Call the Bedrock service to get the response
    response = call_bedrock(message_list=message_list,system_prompts=rag_system_message,extract_filter=False)
    
    # Return the generated text response
    return response['output']['message']['content'][0]['text']