from pathlib import Path
import hmac
import sys
from uuid import uuid4

import streamlit as st


USERS_FILE = Path(__file__).with_name("users.txt")
SRC_DIR = Path(__file__).with_name("src")
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
		st.header("Admin screen")
		st.info("Welcome to the administrator area.")
		st.write("Admin tools can be added here.")
	else:
		show_chat_workspace(username)


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
	return str(content)


def show_chat_workspace(username):
	from modules.chatMemory import get_session_message_history

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
			key=f"chat_selector_{username}",
		)
		if active_id != selected_session["id"]:
			st.session_state[f"active_chat_{username}"] = active_id
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
					response = get_chat_chain().invoke(
						{"question": question},
						config={"configurable": {"session_id": selected_session["id"]}},
					)
					st.markdown(get_message_text(response))
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
