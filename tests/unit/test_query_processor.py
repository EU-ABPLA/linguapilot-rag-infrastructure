import pytest

from core.query_engine.query_processor import QueryProcessor
from core.trace.trace_context import TraceContext


def test_query_processor_outputs_non_empty_keywords_and_dict_filters() -> None:
    processor = QueryProcessor()
    processed = processor.process("What is hybrid retrieval pipeline design")
    assert processed.keywords
    assert isinstance(processed.filters, dict)
    assert processed.filters == {}


def test_query_processor_removes_stopwords_and_deduplicates_keywords() -> None:
    processor = QueryProcessor()
    processed = processor.process("the retrieval and Retrieval pipeline with pipeline")
    assert processed.keywords == ["retrieval", "pipeline"]


def test_query_processor_merges_inline_filters_and_explicit_filters() -> None:
    processor = QueryProcessor()
    processed = processor.process(
        "find grammar rules collection:english lang:en",
        filters={"lang": "zh", "doc_type": "pdf"},
    )
    assert processed.filters["collection"] == "english"
    assert processed.filters["lang"] == "zh"
    assert processed.filters["doc_type"] == "pdf"
    assert "grammar" in processed.keywords
    assert "rules" in processed.keywords


def test_query_processor_fallback_keeps_keywords_non_empty() -> None:
    processor = QueryProcessor(stop_words={"the", "and", "for", "with"})
    processed = processor.process("the and for with")
    assert processed.keywords == ["the", "and", "for", "with"]


def test_query_processor_records_trace_stage() -> None:
    processor = QueryProcessor()
    trace = TraceContext(trace_type="query")
    processor.process("find retrieval strategy", trace=trace)
    stages = [stage for stage in trace.stages if stage["stage"] == "query_processor"]
    assert len(stages) == 1
    assert stages[0]["details"]["keyword_count"] >= 1


def test_query_processor_raises_for_invalid_query_type() -> None:
    processor = QueryProcessor()
    with pytest.raises(ValueError) as exc_info:
        processor.process(123)  # type: ignore[arg-type]
    assert "query must be a string" in str(exc_info.value)
