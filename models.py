from pydantic import BaseModel


class IngestResponse(BaseModel):
    task_id: str
    status: str
    collection_name: str


class IngestStatusResponse(BaseModel):
    task_id: str
    status: str
    chunk_count: int = 0
    error: str = ""


class QueryRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    collection_name: str


class EvalRecord(BaseModel):
    question: str
    answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    timestamp: str = ""
