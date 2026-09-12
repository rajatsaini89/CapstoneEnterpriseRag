import os
import warnings
from math import nan

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import  LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
import utils.EvaluationQuestionUtility as EvaluationQuestionUtility
from utils.ConfigUtility import ConfigUtility
from modules.RagChain import build_chain
from models.EvaluationResult import EvaluationResult

# These metric imports still work with evaluate() on RAGAS 0.4.x.
# Collections API is preferred for future 1.0, but evaluate() is clearer for teaching.
warnings.filterwarnings(
    "ignore",
    message="Importing .* from 'ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)

from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

def evaluateRag():
    try:
        evalUtility = EvaluationQuestionUtility.EvaluationQuestionUtility()
        configUtil = ConfigUtility()
        hybridRetrieverConfig = configUtil.getHybridWeightConfig()
        ragChain = build_chain()
        evaluation_questions = evalUtility.get_all()
    except Exception as error:
        raise RuntimeError(
            "Unable to initialize the RAG evaluation"
        ) from error

    if not evaluation_questions:
        raise ValueError("At least one evaluation question is required")

    print("=" * 60)
    print("Local RAG + RAGAS Evaluation")
    print("=" * 60)
    print(f"Number of evaluation questions: {len(evaluation_questions)}")
    print(f"Retriever k (top chunks): {hybridRetrieverConfig.top_k}")
    print()

   
    print("Running RAG on evaluation questions...")
    rows = []
    question_ids = []

    for i, item in enumerate(evaluation_questions, start=1):
        question_id_for_error = item.get("Id", "unknown") if isinstance(item, dict) else "unknown"
        try:
            question_id = item["Id"]
            question = item["Question"]
            ground_truth = item["Ground_truth"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                raise ValueError("Ground truth must be a non-empty string")

            question_ids.append(question_id)
            print(f" [{i}/{len(evaluation_questions)}] {question}")

            rag_result = ragChain.invoke(
                input={
                    "question": question,
                    "session_id": f"evaluation_session_{i}",
                }
            )
            if "retrieved_contexts" not in rag_result or "answer" not in rag_result:
                raise ValueError(
                    "RAG response is missing 'retrieved_contexts' or 'answer'"
                )

            rows.append(
                {
                    "user_input": question,
                    "retrieved_contexts": rag_result["retrieved_contexts"],
                    "response": rag_result["answer"],
                    "reference": ground_truth,
                }
            )
        except Exception as error:
            raise RuntimeError(
                f"Unable to evaluate question {i} (ID: {question_id_for_error})"
            ) from error

    # ------------------------------------------------------------
    # 3) Create a RAGAS EvaluationDataset
    # ------------------------------------------------------------
    try:
        evaluation_dataset = EvaluationDataset.from_list(rows)
    except Exception as error:
        raise RuntimeError("Unable to create the RAGAS evaluation dataset") from error

    # ------------------------------------------------------------
    # 4) Configure the judge LLM / embeddings used by RAGAS
    # ------------------------------------------------------------
    try:
        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
    except Exception as error:
        raise RuntimeError("Unable to initialize the RAGAS evaluation models") from error

    # ----------------------------------------------
    # 5) Select pedagogically useful RAGAS metrics
    # ----------------------------------------------
    # Generation quality:
    # - Faithfulness              -> is the answer grounded in retrieved context?
    # - ResponseRelevancy         -> does the answer address the question?
    #
    # Retrieval quality:
    # - LLMContextPrecisionWithReference -> are useful chunks ranked well?
    # - LLMContextRecall          -> was needed info retrieved?
    # ----------------------------------------------

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print("\nRunning RAGAS evaluation (this makes additional LLM calls)...")
    try:
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
    except Exception as error:
        raise RuntimeError("RAGAS evaluation failed while calculating metrics") from error

    # RAGAS keeps result rows in the same order as the evaluation dataset.
    # Add the database ID so every score can be traced to its source question.
    try:
        df = result.to_pandas()
        df.insert(0, "question_id", question_ids)

        evaluation_results = [
            EvaluationResult(
                question_id=str(row["question_id"]),
                question=str(row["user_input"]),
                answer=str(row["response"]),
                expected_answer=str(row["reference"]),
                faithfulness=float(row.get("faithfulness", nan)),
                answer_relevancy=float(row.get("answer_relevancy", nan)),
                response_relevancy=float(
                    row.get("response_relevancy", row.get("answer_relevancy", nan))
                ),
                llm_context_precision_with_reference=float(
                    row.get("llm_context_precision_with_reference", nan)
                ),
                context_precision=float(row.get("context_precision", nan)),
                context_recall=float(row.get("context_recall", nan)),
                llm_context_recall=float(
                    row.get("llm_context_recall", row.get("context_recall", nan))
                ),
            )
            for _, row in df.iterrows()
        ]
    except Exception as error:
        raise RuntimeError("Unable to convert RAGAS results into evaluation results") from error

  



    return evaluation_results

