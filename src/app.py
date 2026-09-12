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


def get_error_message(error):
	"""Return the most useful user-facing message from a module exception."""
	message = str(error).strip()
	if message:
		return message

	cause = error.__cause__
	while cause:
		message = str(cause).strip()
		if message:
			return message
		cause = cause.__cause__

	return "An unexpected error occurred."


def show_operation_error(operation, error):
	st.error(f"{operation}: {get_error_message(error)}")


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
		show_admin_workspace()
	else:
		show_chat_workspace(username)


def show_admin_workspace():
	from utils.EvaluationQuestionUtility import EvaluationQuestionUtility

	try:
		questions = EvaluationQuestionUtility().get_all()
	except Exception as error:
		show_operation_error("The evaluation questions could not be loaded", error)
		return
	if "admin_section" not in st.session_state:
		st.session_state["admin_section"] = "evaluation"

	with st.sidebar:
		st.subheader("Admin")
		if st.button(
			"Evaluation dashboard",
			key="evaluation_dashboard_tab",
			type="primary" if st.session_state["admin_section"] == "evaluation" else "secondary",
			icon=":material/monitoring:",
			width="stretch",
		):
			st.session_state["admin_section"] = "evaluation"
			st.rerun()
		if st.button(
			"Question management",
			key="question_management_tab",
			type="primary" if st.session_state["admin_section"] == "questions" else "secondary",
			icon=":material/quiz:",
			width="stretch",
		):
			st.session_state["admin_section"] = "questions"
			st.rerun()
		if st.button(
			"Manage documents",
			key="document_upload_tab",
			type="primary" if st.session_state["admin_section"] == "documents" else "secondary",
			icon=":material/folder_managed:",
			width="stretch",
		):
			st.session_state["admin_section"] = "documents"
			st.rerun()
		if st.button(
			"Configuration",
			key="configuration_tab",
			type="primary" if st.session_state["admin_section"] == "configuration" else "secondary",
			icon=":material/tune:",
			width="stretch",
		):
			st.session_state["admin_section"] = "configuration"
			st.rerun()

	if st.session_state["admin_section"] == "evaluation":
		show_llm_evaluation_dashboard(questions)
	elif st.session_state["admin_section"] == "questions":
		show_evaluation_question_manager(questions)
	elif st.session_state["admin_section"] == "documents":
		show_document_manager()
	else:
		show_configuration_manager()


def show_configuration_manager():
	from utils.ConfigUtility import ConfigUtility

	try:
		config = ConfigUtility()
		values = config.configDatabase.get_all()
	except Exception as error:
		show_operation_error("The configuration could not be loaded", error)
		return

	st.title("Configuration")
	st.caption("Manage the RAG pipeline settings used by the application.")

	with st.container(border=True):
		with st.form("configuration_form"):
			st.subheader("Model settings")
			embedding_provider = st.selectbox(
				"Embedding provider",
				options=["openai", "gemini"],
				index=["openai", "gemini"].index(values["embedding_provider"]),
			)
			llm_provider = st.selectbox(
				"LLM provider",
				options=["openai", "gemini"],
				index=["openai", "gemini"].index(values["llm_provider"]),
			)
			llm_model = st.text_input("LLM model", value=values["llm_model"])
			llm_temperature = st.number_input(
				"LLM temperature",
				min_value=0.0,
				max_value=2.0,
				step=0.1,
				value=float(values["llm_temperature"]),
			)

			st.subheader("Text splitting")
			chunk_size = st.number_input(
				"Chunk size",
				min_value=1,
				step=1,
				value=int(values["chunk_size"]),
			)
			min_chunk_size = st.number_input(
				"Minimum chunk size",
				min_value=1,
				step=1,
				value=int(values["min_chunk_size"]),
			)
			chunk_overlap = st.number_input(
				"Chunk overlap",
				min_value=0,
				step=1,
				value=int(values["chunk_overlap"]),
			)
			st.subheader("Hybrid retriever")
			hybrid_config = values["hybridRetrieverConfig"]
			bm25_weight = st.number_input(
				"BM25 weight",
				min_value=0.0,
				max_value=1.0,
				step=0.05,
				value=float(hybrid_config["bm25_weight"]),
			)
			embedding_weight = st.number_input(
				"Embedding weight",
				min_value=0.0,
				max_value=1.0,
				step=0.05,
				value=float(hybrid_config["embedding_weight"]),
			)
			hybrid_top_k = st.number_input(
				"Hybrid top K",
				min_value=1,
				step=1,
				value=int(hybrid_config["top_k"]),
			)

			save_submitted = st.form_submit_button(
				"Save configuration",
				type="primary",
				icon=":material/save:",
			)

	if not save_submitted:
		return

	if not llm_model.strip():
		st.error("LLM model is required.")
		return
	if min_chunk_size > chunk_size:
		st.error("Minimum chunk size cannot be larger than chunk size.")
		return
	if chunk_overlap >= chunk_size:
		st.error("Chunk overlap must be smaller than chunk size.")
		return
	if abs(bm25_weight + embedding_weight - 1.0) > 1e-9:
		st.error("BM25 and embedding weights must sum to 1.")
		return

	updated_values = {
		"embedding_provider": embedding_provider,
		"llm_provider": llm_provider,
		"llm_model": llm_model.strip(),
		"llm_temperature": llm_temperature,
		"chunk_size": chunk_size,
		"min_chunk_size": min_chunk_size,
		"chunk_overlap": chunk_overlap,
		"hybridRetrieverConfig": {
			"bm25_weight": bm25_weight,
			"embedding_weight": embedding_weight,
			"top_k": hybrid_top_k,
		},
	}
	for key, value in updated_values.items():
		try:
			if not config.configDatabase.update(key, value):
				config.configDatabase.create(key, value)
		except Exception as error:
			show_operation_error("The configuration could not be saved", error)
			return
	get_chat_chain.clear()
	st.success("Configuration saved. New chat requests will use the updated settings.")


def show_document_manager():
	from modules.docReader import read_docx, read_pdf, read_txt
	from modules.textSplitter import split_documents
	from modules.VectorStore import addDocsToVectorStore
	from utils.vectorStoreUtility import initializeVectorStore

	docs_folder = Path(__file__).resolve().parents[1] / "Docs"

	st.title("Manage documents")
	st.caption("View, upload, and delete documents in the knowledge base.")

	with st.container(border=True):
		st.subheader("Documents in Docs")
		if not docs_folder.exists():
			st.info("The Docs folder does not contain any documents yet.")
		else:
			documents = sorted(
				(path for path in docs_folder.iterdir() if path.is_file()),
				key=lambda path: path.name.lower(),
			)
			if not documents:
				st.warning("At least one document must be present in the Docs folder.")
			else:
				if len(documents) == 1:
					st.info("At least one document must remain in the Docs folder, so the last document cannot be deleted.")
				mime_types = {
					".pdf": "application/pdf",
					".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
					".txt": "text/plain",
				}
				for document_path in documents:
					with st.container(horizontal=True, vertical_alignment="center"):
						st.write(document_path.name)
						st.caption(f"{document_path.stat().st_size:,} bytes")
						st.download_button(
							"Download",
							data=document_path.read_bytes(),
							file_name=document_path.name,
							mime=mime_types.get(document_path.suffix.lower(), "application/octet-stream"),
							key=f"download_document_{document_path.name}",
							icon=":material/download:",
						)
						delete_submitted = st.button(
							"Delete",
							key=f"delete_document_{document_path.name}",
							icon=":material/delete:",
							disabled=len(documents) == 1,
						)

					if delete_submitted:
						if len(documents) == 1:
							st.error("At least one document must remain in the Docs folder.")
							return
						try:
							document_path.unlink()
							initializeVectorStore(forceRecreate=True)
							get_chat_chain.clear()
						except Exception as error:
							show_operation_error("The document could not be deleted", error)
						else:
							st.success(f"'{document_path.name}' was deleted.")
							st.rerun()

	with st.container(border=True):
		st.subheader("Recreate vector store")
		st.caption("Delete the current FAISS index and rebuild it from every document in Docs.")
		recreate_submitted = st.button(
			"Recreate vector store",
			icon=":material/refresh:",
		)

	if recreate_submitted:
		with st.spinner("Recreating vector store from documents in Docs..."):
			try:
				initializeVectorStore(forceRecreate=True)
				get_chat_chain.clear()
			except Exception as error:
				show_operation_error("The vector store could not be recreated", error)
			else:
				st.success("The vector store was recreated from the documents in Docs.")

	with st.container(border=True):
		uploaded_file = st.file_uploader(
			"Choose a document",
			type=["pdf", "docx", "txt"],
			key="knowledge_base_upload",
		)
		add_submitted = st.button(
			"Add to knowledge base",
			type="primary",
			icon=":material/upload_file:",
			disabled=uploaded_file is None,
		)

	if not add_submitted or uploaded_file is None:
		return

	file_name = Path(uploaded_file.name).name
	file_path = docs_folder / file_name
	if file_path.exists():
		st.error(f"A document named '{file_name}' already exists in Docs.")
		return

	docs_folder.mkdir(parents=True, exist_ok=True)
	file_path.write_bytes(uploaded_file.getvalue())

	try:
		suffix = file_path.suffix.lower()
		if suffix == ".pdf":
			documents = read_pdf(str(file_path))
		elif suffix == ".docx":
			documents = read_docx(str(file_path))
		else:
			documents = read_txt(str(file_path))

		if not documents:
			file_path.unlink()
			st.error("The uploaded document did not contain any readable text.")
			return

		addDocsToVectorStore(split_documents(documents))
		get_chat_chain.clear()
	except Exception as error:
		file_path.unlink(missing_ok=True)
		show_operation_error("The document could not be added", error)
		return

	st.success(f"'{file_name}' was added to Docs and the vector store.")


def show_evaluation_question_manager(questions):
	from utils.EvaluationQuestionUtility import EvaluationQuestionUtility

	st.title("Evaluation questions")
	st.caption("Create, update, and remove the questions used to evaluate the RAG system.")
	question_utility = EvaluationQuestionUtility()

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
				try:
					question_utility.create(question.strip(), ground_truth.strip())
				except Exception as error:
					show_operation_error("The evaluation question could not be added", error)
					return
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
				try:
					updated = question_utility.update(
						selected_id,
						updated_question.strip(),
						updated_ground_truth.strip(),
					)
				except Exception as error:
					show_operation_error("The evaluation question could not be updated", error)
					return
				if not updated:
					st.error("The selected evaluation question no longer exists.")
					return
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
			try:
				deleted = question_utility.delete(selected_id)
			except Exception as error:
				show_operation_error("The evaluation question could not be deleted", error)
				return
			if not deleted:
				st.error("The selected evaluation question no longer exists.")
				return
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


def show_llm_evaluation_dashboard(questions):
	"""Run the configured RAGAS evaluation and display its results for admins."""
	with st.container(border=True):
		st.subheader("LLM evaluation")
		st.caption("Run the evaluation set against the current RAG pipeline and review its scores.")

		run_evaluation = st.button(
			"Run evaluation",
			 type="primary",
			 icon=":material/monitoring:",
			 disabled=not questions,
		)

		if run_evaluation:
			with st.spinner("Running RAGAS evaluation. This may take a few minutes..."):
				try:
					from modules.RagEvaluator import evaluateRag

					st.session_state["llm_evaluation_results"] = evaluateRag()
					st.session_state.pop("llm_evaluation_error", None)
				except Exception as error:
					st.session_state["llm_evaluation_error"] = str(error)
					st.session_state.pop("llm_evaluation_results", None)

		if not questions:
			st.info("Add at least one evaluation question before running the evaluation.")
			return

		if st.session_state.get("llm_evaluation_error"):
			show_operation_error(
				"The evaluation could not be completed",
				RuntimeError(st.session_state["llm_evaluation_error"]),
			)

		results = st.session_state.get("llm_evaluation_results")
		if not results:
			st.info("No evaluation run yet. Select 'Run evaluation' to generate a report.")
			return

		import pandas as pd

		metric_columns = {
			"faithfulness": "Faithfulness",
			"response_relevancy": "Response relevancy",
			"llm_context_precision_with_reference": "Context precision",
			"llm_context_recall": "Context recall",
		}
		rows = [
			{
				"Question ID": result.question_id,
				"Question": result.question,
				"Expected answer": result.expected_answer,
				"LLM answer": result.answer,
				**{
					label: getattr(result, field)
					for field, label in metric_columns.items()
				},
			}
			for result in results
		]
		results_df = pd.DataFrame(rows)
		average_scores = results_df.drop(columns=["Question ID"]).mean(numeric_only=True)

		with st.container(horizontal=True):
			st.metric("Questions evaluated", len(results), border=True)
			st.metric("Average Context Precision", f"{average_scores['Context precision']:.2f}", border=True)
			st.metric("Average Context Recall", f"{average_scores['Context recall']:.2f}", border=True)
			st.metric("Average Faithfulness", f"{average_scores['Faithfulness']:.2f}", border=True)
			st.metric("Average Response Relevancy", f"{average_scores['Response relevancy']:.2f}", border=True)

		chart_df = average_scores.rename("Average score").to_frame()
		with st.container(border=True):
			st.subheader("Average metric scores")
			st.bar_chart(chart_df, y="Average score", x_label="Metric", y_label="Score")

		with st.container(border=True):
			st.subheader("Question-level results")
			results_table = results_df.style.set_properties(
				subset=["Question", "Expected answer", "LLM answer"],
				**{"white-space": "pre-wrap", "word-wrap": "break-word"},
			)
			st.dataframe(
				results_table,
				column_config={
					"Question ID": st.column_config.TextColumn("Question ID"),
					"Question": st.column_config.TextColumn(
						"Question",
						width="large",
					),
					"Expected answer": st.column_config.TextColumn(
						"Expected answer",
						width="large",
					),
					"LLM answer": st.column_config.TextColumn(
						"LLM answer",
						width="large",
					),
					**{
						label: st.column_config.NumberColumn(label, format="%.2f")
						for label in metric_columns.values()
					},
				},
				hide_index=True,
			)


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
	sources = get_structured_response_field(response, "sources")
	if sources is None:
		sources = getattr(response, "additional_kwargs", {}).get("sources", [])
	if not sources:
		return

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
		if st.button("+ New chat", width="stretch"):
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
			try:
				delete_session_memory(deleted_id)
			except Exception as error:
				show_operation_error("The chat could not be deleted", error)
				return
			sessions[:] = [session for session in sessions if session["id"] != deleted_id]

			if not sessions:
				sessions.append(create_chat_session(username, 0))
			st.session_state[f"active_chat_{username}"] = sessions[0]["id"]
			st.rerun()

	st.header(selected_session["name"])
	st.caption(f"Signed in as {username}")

	try:
		history = get_session_message_history(selected_session["id"])
	except Exception as error:
		show_operation_error("Chat history could not be loaded", error)
		return
	for message in history.messages:
		if message.type in ("human", "ai"):
			with st.chat_message("user" if message.type == "human" else "assistant"):
				st.markdown(get_message_text(message))
				if message.type == "ai":
					show_response_sources(message)

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
					if history.messages and history.messages[-1].type == "ai":
						sources = get_structured_response_field(response, "sources") or []
						history.messages[-1].additional_kwargs["sources"] = sources
					st.rerun()
				except Exception as error:
					show_operation_error("Unable to get a response", error)


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
