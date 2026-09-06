from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from modules.chatMemory import getMemorySummary, getRecentContext
from utils.ConfigUtility import ConfigUtility 
from modules.HybridRetriever import retrieve_documents


def build_chain():
    config = ConfigUtility()
    
    llm_provider = config.getLLMProvider().lower()
    llm_model = config.getLLMModel()
    llm_temperature = config.getLLMTemperature()

    print(f"Building chain with LLM provider: {llm_provider}, model: {llm_model}")

    if llm_provider == "openai":
        llm = ChatOpenAI(model_name=llm_model, temperature=llm_temperature , api_key=config.getOpenAIKey())
    elif llm_provider in ["google", "gemini"]:
        llm = ChatGoogleGenerativeAI(model_name=llm_model, temperature=llm_temperature, google_api_key=config.getGeminiKey())

    system_prompt = """
    You are a helpful RAG (Retrieval-Augmented Generation) assistant for a well known enterprise organization.
      You will be provided with a user question
      You will be provide with relevant memory of two types
       1. A summary of past conversations.
       2. The most recent few question-answer turns for short-term context.
       
      You will also be provided with relvant chunks for documents.
      Use this information to generate a concise and accurate response to the user's question with following constraints, answers should be in a professional and human tone, and should be concise and accurate.
      Constraints:
      You are to strictly allowed to answer questions based on the provided information as part of retrived context.
       If the retrieved context do not have enough information inform the user about lack of sufficient information.
      """

    prompt = ChatPromptTemplate.from_template("""
    SystemPrompt:
    {system_prompt}

    Converstion Summary History:
    {memory_summary}

    Recent Chat History:
    {recent_memory}

    Retrived Context:
    {context}

    User Question:
    {question}

    If the user asks to summarize or refer to earlier parts of the conversation, rely primarily on the chat history and memory summary.
    Otherwise, combine retrieved context and memories to answer effectively.
    """)

    def combine_docs(docs)->str:
        if not docs:
            return "No relevant documents retrieved."
        return "\n\n".join(doc[0].page_content for doc in docs)

    rag_chain=(RunnableParallel(
        {
            "system_prompt" : RunnableLambda(lambda _:system_prompt),
            "memory_summary": RunnableLambda(lambda x: getMemorySummary(x["session_id"]) ),
            "recent_memory": RunnableLambda(lambda x: getRecentContext(x["session_id"]) ),
            "question": RunnableLambda(lambda x:x["question"] ),
            "context":  RunnableLambda(lambda x: x["question"]) | RunnableLambda(retrieve_documents) |  RunnableLambda(combine_docs),
        }
    )) | prompt | llm

   

    print(f"Hybrid RAG chain built successfully with provider: {llm_provider}")
    return rag_chain
   