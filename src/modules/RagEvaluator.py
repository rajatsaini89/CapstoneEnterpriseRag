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
    evalUtility = EvaluationQuestionUtility.EvaluationQuestionUtility()
    configUtil = ConfigUtility()
    hybridRetrieverConfig = configUtil.getHybridWeightConfig()
    ragChain = build_chain()
    EVALUATION_QUESTIONS = evalUtility.get_all()

    print("=" * 60)
    print("Local RAG + RAGAS Evaluation")
    print("=" * 60)
    print(f"Number of evaluation questions: {len(EVALUATION_QUESTIONS)}")
    print(f"Retriever k (top chunks): {hybridRetrieverConfig.top_k}")
    print()

   
    print("Running RAG on evaluation questions...")
    rows = []
    question_ids = []

    for i, item in enumerate(EVALUATION_QUESTIONS, start=1):
        question_ids.append(item["Id"])
        question = item["Question"]
        ground_truth = item["Ground_truth"]

        print(f" [{i}/{len(EVALUATION_QUESTIONS)}] {question}")

        rag_result = ragChain.invoke(input={"question": question, "session_id": f"evaluation_session_{i}"})

        # Field names required by the current RAGAS EvaluationDataset API
        rows.append(
            {
                "user_input": question,
                "retrieved_contexts": rag_result["retrieved_contexts"],
                "response": rag_result["answer"],
                "reference": ground_truth,
            }
        )

    # ------------------------------------------------------------
    # 3) Create a RAGAS EvaluationDataset
    # ------------------------------------------------------------
    evaluation_dataset = EvaluationDataset.from_list(rows)

    # ------------------------------------------------------------
    # 4) Configure the judge LLM / embeddings used by RAGAS
    # ------------------------------------------------------------
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

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
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # RAGAS keeps result rows in the same order as the evaluation dataset.
    # Add the database ID so every score can be traced to its source question.
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

  



    return evaluation_results

