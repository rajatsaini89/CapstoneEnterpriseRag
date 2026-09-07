from pathlib import Path
import hmac
import sys
from uuid import uuid4

import streamlit as st
import dotenv 

dotenv.load_dotenv()

USERS_FILE = Path(__file__).with_name("users.txt")
SRC_DIR = Path(__file__)
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))


def load_users():
	"""Load users from username|password|role records."""
	
	users = {}

	if not USERS_FILE.exists():
		return users

	for line in USERS_FILE.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue

		fields = [field.strip() for field in line.split("|")]
		if len(fields) == 3 and all(fields):
			username, password, role = fields
			users[username] = {"password": password, "role": role.lower()}

	return users


def authenticate(username, password, users):
	print(f"Authenticating user: {username},{password}")
	user = users.get(username)
	if user and hmac.compare_digest(user["password"], password):
		return user
	return None


def show_login(users):
	st.title("Welcome")
	st.subheader("Sign in to continue")

	with st.form("login_form"):
		username = st.text_input("Username")
		password = st.text_input("Password", type="password")
		submitted = st.form_submit_button("Log in", type="primary")

	if submitted:
		user = authenticate(username.strip(), password, users)
		if user:
			st.session_state["authenticated"] = True
			st.session_state["username"] = username.strip()
			st.session_state["role"] = user["role"]
			st.rerun()
		else:
			st.error("Invalid username or password.")


def show_authenticated_page():
	username = st.session_state["username"]
	role = st.session_state["role"]

	with st.sidebar:
		st.write(f"Signed in as **{username}**")
		if st.button("Log out"):
			for key in ("authenticated", "username", "role"):
				st.session_state.pop(key, None)
			st.rerun()

	st.title("Welcome!")
	st.write(f"Hello, **{username}**. You are signed in successfully.")

	if role == "admin":
		show_evaluation_question_manager()
	else:
		show_chat_workspace(username)


def show_evaluation_question_manager():
	from utils.EvaluationQuestionUtility import EvaluationQuestionUtility

	st.title("Evaluation questions")
	st.caption("Create, update, and remove the questions used to evaluate the RAG system.")
	question_utility = EvaluationQuestionUtility()
	questions = question_utility.get_all()

	with st.container(border=True):
		st.subheader("Add a question")
		with st.form("create_evaluation_question_form", clear_on_submit=True):
			question = st.text_area("Question", placeholder="Enter the evaluation question")
			ground_truth = st.text_area(
				"Ground truth",
				placeholder="Enter the expected answer",
			)
			create_submitted = st.form_submit_button(
				"Add question",
				type="primary",
				icon=":material/add:",
			)

		if create_submitted:
			if not question.strip() or not ground_truth.strip():
				st.error("Question and ground truth are required.")
			else:
				question_utility.create(question.strip(), ground_truth.strip())
				st.success("Evaluation question added.")
				st.rerun()

	if questions:
		st.subheader("Manage existing questions")
		question_options = {item["Id"]: item for item in questions}
		selected_id = st.selectbox(
			"Question",
			options=list(question_options),
			format_func=lambda question_id: (
				f"{question_id}: {question_options[question_id]['Question']}"
			),
			key="selected_evaluation_question_id",
		)
		selected_question = question_options[selected_id]

		with st.form("edit_evaluation_question_form"):
			updated_question = st.text_area(
				"Question text",
				value=selected_question["Question"],
				key=f"question_text_{selected_id}",
			)
			updated_ground_truth = st.text_area(
				"Ground truth",
				value=selected_question["Ground_truth"],
				key=f"ground_truth_{selected_id}",
			)
			update_submitted = st.form_submit_button(
				"Save changes",
				type="primary",
				icon=":material/save:",
			)

		if update_submitted:
			if not updated_question.strip() or not updated_ground_truth.strip():
				st.error("Question and ground truth are required.")
			else:
				question_utility.update(
					selected_id,
					updated_question.strip(),
					updated_ground_truth.strip(),
				)
				st.success("Evaluation question updated.")
				st.rerun()

		with st.container(horizontal=True):
			st.caption(f"Question ID: {selected_id}")
			delete_submitted = st.button(
				"Delete question",
				icon=":material/delete:",
				type="secondary",
			)

		if delete_submitted:
			question_utility.delete(selected_id)
			st.success("Evaluation question deleted.")
			st.rerun()

		st.subheader("All questions")
		st.dataframe(
			questions,
			column_config={
				"Id": st.column_config.NumberColumn("ID", width="small"),
				"Question": st.column_config.TextColumn("Question", width="large"),
				"Ground_truth": st.column_config.TextColumn("Ground truth", width="large"),
			},
			hide_index=True,
		)
	else:
		st.info("No evaluation questions have been added yet.")


def create_chat_session(username, session_count):
	return {
		"id": f"{username}:{uuid4().hex}",
		"name": f"Chat {session_count + 1}",
	}


def get_chat_sessions(username):
	key = f"chat_sessions_{username}"
	if key not in st.session_state:
		st.session_state[key] = [create_chat_session(username, 0)]
	return st.session_state[key]


@st.cache_resource
def get_chat_chain():
	from modules.RagChain import build_chain
	from modules.chatMemory import llm_chain_with_memory

	return llm_chain_with_memory(build_chain())


def get_message_text(message):
	content = getattr(message, "content", message)
	if isinstance(content, list):
		return "\n".join(
			item.get("text", str(item)) if isinstance(item, dict) else str(item)
			for item in content
		)
	if isinstance(content, dict) and "answer" in content:
		return str(content["answer"])
	return str(content)


def get_structured_response_field(response, field, default=None):
	if isinstance(response, dict):
		return response.get(field, default)
	return getattr(response, field, default)


def show_response_sources(response):
	sources = get_structured_response_field(response, "sources", []) or []
	if sources:
		with st.expander("Sources"):
			for source in sources:
				st.markdown(f"- {source}")


def show_chat_workspace(username):
	from modules.chatMemory import delete_session_memory, get_session_message_history

	sessions = get_chat_sessions(username)
	selected_id = st.session_state.get(f"active_chat_{username}", sessions[0]["id"])
	selected_session = next(
		(session for session in sessions if session["id"] == selected_id), sessions[0]
	)
	st.session_state[f"active_chat_{username}"] = selected_session["id"]

	with st.sidebar:
		st.subheader("Your chats")
		if st.button("+ New chat", use_container_width=True):
			new_session = create_chat_session(username, len(sessions))
			sessions.append(new_session)
			st.session_state[f"active_chat_{username}"] = new_session["id"]
			st.rerun()

		labels = {session["id"]: session["name"] for session in sessions}
		active_id = st.radio(
			"Sessions",
			options=list(labels),
			format_func=labels.get,
			index=list(labels).index(selected_session["id"]),
			key=f"chat_selector_{username}_{selected_session['id']}",
		)
		if active_id != selected_session["id"]:
			st.session_state[f"active_chat_{username}"] = active_id
			st.rerun()

		if st.button("Delete chat", icon=":material/delete:"):
			deleted_id = selected_session["id"]
			sessions[:] = [session for session in sessions if session["id"] != deleted_id]
			delete_session_memory(deleted_id)

			if not sessions:
				sessions.append(create_chat_session(username, 0))
			st.session_state[f"active_chat_{username}"] = sessions[0]["id"]
			st.rerun()

	st.header(selected_session["name"])
	st.caption(f"Signed in as {username}")

	history = get_session_message_history(selected_session["id"])
	for message in history.messages:
		if message.type in ("human", "ai"):
			with st.chat_message("user" if message.type == "human" else "assistant"):
				st.markdown(get_message_text(message))

	question = st.chat_input("Ask a question")
	if question:
		with st.chat_message("user"):
			st.markdown(question)
		with st.chat_message("assistant"):
			with st.spinner("Thinking..."):
				try:
					print(f"Invoking chat chain with question: {question}")
					response = get_chat_chain().invoke(
						{"question": question},
						config={"configurable": {"session_id": selected_session["id"]}},
					)
					st.markdown(get_message_text(response))
					show_response_sources(response)
					st.rerun()
				except Exception as error:
					st.error(f"Unable to get a response: {error}")


def main():
	st.set_page_config(page_title="Welcome")
	users = load_users()

	if not users:
		st.error(f"No users found. Add accounts to `{USERS_FILE.name}`.")
		st.stop()

	if st.session_state.get("authenticated"):
		show_authenticated_page()
	else:
		show_login(users)


if __name__ == "__main__":
	main()
