import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

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

from evaluation_dataset import EVALUATION_QUESTIONS
from rag import RETRIEVE_K, ask, build_vector_store

load_dotenv()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file before running evaluation."
        )

    print("=" * 60)
    print("Local RAG + RAGAS Evaluation")
    print("=" * 60)
    print(f"Number of evaluation questions: {len(EVALUATION_QUESTIONS)}")
    print(f"Retriever k (top chunks): {RETRIEVE_K}")
    print()

    # ------------------------------------------------------------
    # 1) Make sure the vector store exists
    # ------------------------------------------------------------
    build_vector_store()

    # ------------------------------------------------------------
    # 2) Run each evaluation question through the RAG pipeline
    #    Collect: question, retrieved contexts, answer, ground truth
    # ------------------------------------------------------------
    print("Running RAG on evaluation questions...")
    rows = []

    for i, item in enumerate(EVALUATION_QUESTIONS, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f" [{i}/{len(EVALUATION_QUESTIONS)}] {question}")

        rag_result = ask(question)

        # Field names required by the current RAGAS EvaluationDataset API
        rows.append(
            {
                "user_input": question,
                "retrieved_contexts": rag_result["contexts"],
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

    # ------------------------------------------------------------
    # 6) Display overall scores
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL RAGAS METRICS")
    print("=" * 60)
    print(result)

    # Keep the table readable for classroom projection.
    # Column names can vary slightly across RAGAS versions, so we detect them.
    preferred = [
        "user_input",
        "response",
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "llm_context_precision_with_reference",
        "context_precision",
        "context_recall",
        "llm_context_recall",
    ]

    display_cols = [col for col in preferred if col in df.columns]

    # Always include any remaining numeric metric columns
    for col in df.columns:
        if col not in display_cols and df[col].dtype.kind in "fi":
            display_cols.append(col)

    if not display_cols:
        display_cols = list(df.columns)

        # Always include any remaining numeric metric columns
    for col in df.columns:
        if col not in display_cols and df[col].dtype.kind in "fi":
            display_cols.append(col)

    if not display_cols:
        display_cols = list(df.columns)

    # Truncate long text fields for console readability
    preview = df[display_cols].copy()
    for col in ["user_input", "response"]:
        if col in preview.columns:
            preview[col] = preview[col].astype(str).str.slice(0, 80)

    print(preview.to_string(index=False))

    # ---------------------------------------------
    # 8) Save results for later review
    # ---------------------------------------------
    output_path = "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()