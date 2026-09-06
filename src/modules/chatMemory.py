from langchain.schema.runnable import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from utils.ConfigUtility import ConfigUtility

SESSION_MESSAGE_HISTORY = {}
SESSION_SUMMARY_MEMORY = {}
# SESSION_RECENT_MEMORY = {}

configUtil = ConfigUtility()

def get_session_message_history(session_id: str):
    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()
    return SESSION_MESSAGE_HISTORY[session_id]


def delete_session_memory(session_id: str):
    SESSION_MESSAGE_HISTORY.pop(session_id, None)
    SESSION_SUMMARY_MEMORY.pop(session_id, None)


def getSessionSummaryMemory(session_id: str):
    if session_id not in SESSION_SUMMARY_MEMORY:
        providerSettings = configUtil.getMemoryProviderConfig()
        SESSION_SUMMARY_MEMORY[session_id] = ConversationSummaryMemory(
            llm=ChatOpenAI(model_name=providerSettings.llm_model, temperature=0),
            chat_memory=get_session_message_history(session_id),
            return_messages=True,
            
        )
    return SESSION_SUMMARY_MEMORY[session_id]

def getMemorySummary(session_id: str):
    summary_memory = getSessionSummaryMemory(session_id)
    return summary_memory.load_memory_variables({})["history"]

def getRecentContext(session_id: str):
    return getFormatedRecentContext(session_id, configUtil.getRecentMemoryTopK())

def getFormatedRecentContext(session_id: str, k: int):
    recent_messages = get_session_message_history(session_id).messages

    if len(recent_messages) > k*2:
        recent_messages = recent_messages[-k*2:]
    elif len(recent_messages) == 0:
        return "No recent context available."

    formatted_messages = []
    for message in recent_messages:
        formatted_messages.append(f"{message.type}: {message.content}")
    return "\n".join(formatted_messages)

def update_memory(session_id: str, user_input: str, ai_output: str):
    history = get_session_message_history(session_id)
    history.add_user_message(user_input)
    history.add_ai_message(ai_output)

    

def llm_chain_with_memory(chain,  enabled:bool = True, recent_k:int = 2):
    if not enabled:
        print("Memory is disabled for this chain.")
        return chain

   
    def chain_with_summary(input_data, config):
        
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        print(f"Adding memory to chain for session: {session_id}")  
        summary_memory = getSessionSummaryMemory(session_id)


        
        response = chain.invoke({"question":f"{input_data.get('question')}?",
                        "session_id": config.get("configurable", {}).get("session_id")}               )
        user_input = input_data.get("question")
        ai_output = getattr(response, "content",str(response))


        # update_memory(session_id, user_input, ai_output)

        messages = summary_memory.chat_memory.messages
        existing_summary =summary_memory.load_memory_variables({}).get("summary", "")
        new_summary = summary_memory.predict_new_summary(messages, existing_summary)
        summary_memory._buffer = new_summary

        return response
    
    runnable_with_summary = RunnableLambda(chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda config: get_session_message_history(config),
        input_messages_key="question",
        output_messages_key=None,
        history_messages_key="history"
    )