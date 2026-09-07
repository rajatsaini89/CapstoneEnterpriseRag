class EvaluationResult:
    def __init__(self, question_id: str, faithfulness: float, answer_relevancy: float, response_relevancy: float, llm_context_precision_with_reference: float, context_precision: float, context_recall: float, llm_context_recall: float):
        self.question_id = question_id
        self.faithfulness = faithfulness
        self.answer_relevancy = answer_relevancy
        self.response_relevancy = response_relevancy
        self.llm_context_precision_with_reference = llm_context_precision_with_reference
        self.context_precision = context_precision
        self.context_recall = context_recall
        self.llm_context_recall = llm_context_recall