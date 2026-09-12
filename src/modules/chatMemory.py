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
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A non-empty session ID is required to access message history")

    try:
        if session_id not in SESSION_MESSAGE_HISTORY:
            SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()
        return SESSION_MESSAGE_HISTORY[session_id]
    except Exception as error:
        raise RuntimeError(
            f"Unable to load message history for session '{session_id}'"
        ) from error


def delete_session_memory(session_id: str):
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A non-empty session ID is required to delete memory")

    try:
        SESSION_MESSAGE_HISTORY.pop(session_id, None)
        SESSION_SUMMARY_MEMORY.pop(session_id, None)
    except Exception as error:
        raise RuntimeError(
            f"Unable to delete memory for session '{session_id}'"
        ) from error


def getSessionSummaryMemory(session_id: str):
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A non-empty session ID is required to access summary memory")

    try:
        if session_id not in SESSION_SUMMARY_MEMORY:
            providerSettings = configUtil.getMemoryProviderConfig()
            SESSION_SUMMARY_MEMORY[session_id] = ConversationSummaryMemory(
                llm=ChatOpenAI(model_name=providerSettings.llm_model, temperature=0),
                chat_memory=get_session_message_history(session_id),
                return_messages=True,
            )
        return SESSION_SUMMARY_MEMORY[session_id]
    except Exception as error:
        raise RuntimeError(
            f"Unable to initialize summary memory for session '{session_id}'"
        ) from error

def getMemorySummary(session_id: str):
    try:
        summary_memory = getSessionSummaryMemory(session_id)
        return summary_memory.load_memory_variables({})["history"]
    except Exception as error:
        raise RuntimeError(
            f"Unable to load memory summary for session '{session_id}'"
        ) from error

def getRecentContext(session_id: str):
    try:
        return getFormatedRecentContext(session_id, configUtil.getRecentMemoryTopK())
    except Exception as error:
        raise RuntimeError(
            f"Unable to load recent context for session '{session_id}'"
        ) from error

def getFormatedRecentContext(session_id: str, k: int):
    if not isinstance(k, int) or k < 1:
        raise ValueError("Recent context count must be a positive integer")

    try:
        recent_messages = get_session_message_history(session_id).messages

        if len(recent_messages) > k * 2:
            recent_messages = recent_messages[-k * 2:]
        elif len(recent_messages) == 0:
            return "No recent context available."

        formatted_messages = []
        for message in recent_messages:
            formatted_messages.append(f"{message.type}: {message.content}")
        return "\n".join(formatted_messages)
    except Exception as error:
        raise RuntimeError(
            f"Unable to format recent context for session '{session_id}'"
        ) from error

def update_memory(session_id: str, user_input: str, ai_output: str):
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("User input must be a non-empty string")
    if not isinstance(ai_output, str) or not ai_output.strip():
        raise ValueError("AI output must be a non-empty string")

    try:
        history = get_session_message_history(session_id)
        history.add_user_message(user_input)
        history.add_ai_message(ai_output)
    except Exception as error:
        raise RuntimeError(
            f"Unable to update conversation memory for session '{session_id}'"
        ) from error


def llm_chain_with_memory(chain,  enabled:bool = True, recent_k:int = 2):
    if not enabled:
        print("Memory is disabled for this chain.")
        return chain

   
    def chain_with_summary(input_data, config):
        session_id = "<unknown>"
        try:
            configurable = config.get("configurable", {})
            session_id = configurable.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("A non-empty session ID is required to invoke the memory chain")

            print(f"Adding memory to chain for session: {session_id}")
            summary_memory = getSessionSummaryMemory(session_id)
            response = chain.invoke(
                {
                    "question": f"{input_data.get('question')}?",
                    "session_id": session_id,
                }
            )

            messages = summary_memory.chat_memory.messages
            existing_summary = summary_memory.load_memory_variables({}).get(
                "summary", ""
            )
            new_summary = summary_memory.predict_new_summary(messages, existing_summary)
            summary_memory._buffer = new_summary

            return response
        except Exception as error:
            raise RuntimeError(
                f"Unable to process memory chain for session '{session_id}'"
            ) from error
    
    runnable_with_summary = RunnableLambda(chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda config: get_session_message_history(config),
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="history"
    )