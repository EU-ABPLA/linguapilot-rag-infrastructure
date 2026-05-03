from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

def render_page() -> None:
    import streamlit as st

    st.header("Evaluation Panel")
    from core.query_engine.hybrid_search import HybridSearch
    from core.query_engine.reranker import Reranker
    from core.settings import SettingsError, load_settings
    from libs.evaluator.evaluator_factory import EvaluatorFactory
    from observability.evaluation.eval_runner import EvalRunner

    try:
        settings = load_settings("config/settings.yaml")
    except SettingsError as exc:
        st.error(str(exc))
        return
    default_backends = list(settings.evaluation.backends)
    options = _backend_options(default_backends)
    selected_backends = st.multiselect(
        "Evaluation Backends",
        options=options,
        default=default_backends,
    )
    test_set_path = st.text_input("Golden Test Set Path", value=settings.evaluation.golden_test_set)
    run = st.button("Run Evaluation", disabled=not selected_backends)
    if run:
        with st.spinner("Running evaluation..."):
            try:
                evaluator = EvaluatorFactory.create(
                    {"evaluation": {"backends": list(selected_backends)}}
                )
                hybrid_search = HybridSearch(settings=settings)
                reranker = Reranker(settings=settings)
                runner = EvalRunner(
                    settings=settings,
                    hybrid_search=hybrid_search,
                    evaluator=evaluator,
                    reranker=reranker,
                )
                report = runner.run(test_set_path)
            except Exception as exc:
                st.error(str(exc))
                return
        _push_history(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "backends": ",".join(selected_backends),
                "hit_rate": report.hit_rate,
                "mrr": report.mrr,
                "source_hit_rate": report.source_hit_rate,
                "completed_cases": report.completed_cases,
                "failed_cases": report.failed_cases,
            }
        )
        st.success("Evaluation completed.")
        _render_summary(report)
        _render_cases(report.cases)
    history = _history()
    if history:
        st.subheader("History Compare")
        st.dataframe(history, use_container_width=True)
        st.line_chart(
            history,
            x="timestamp",
            y=["hit_rate", "mrr", "source_hit_rate"],
        )


def _render_summary(report: Any) -> None:
    import streamlit as st

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Hit Rate", _format_float(report.hit_rate))
    col2.metric("MRR", _format_float(report.mrr))
    col3.metric("Source Hit Rate", _format_float(report.source_hit_rate))
    col4, col5, col6 = st.columns(3)
    col4.metric("Total Cases", str(report.total_cases))
    col5.metric("Completed Cases", str(report.completed_cases))
    col6.metric("Failed Cases", str(report.failed_cases))
    if report.avg_metrics:
        rows = []
        for key, value in sorted(report.avg_metrics.items()):
            rows.append({"metric": key, "avg_value": float(value)})
        st.subheader("Average Metrics")
        st.dataframe(rows, use_container_width=True)


def _render_cases(cases: Sequence[Any]) -> None:
    import streamlit as st

    st.subheader("Case Details")
    rows: List[Dict[str, Any]] = []
    for item in cases:
        rows.append(
            {
                "query": item.query,
                "hit": item.hit,
                "mrr": item.mrr,
                "source_hit": item.source_hit,
                "retrieved_chunk_ids": ", ".join(item.retrieved_chunk_ids),
                "expected_chunk_ids": ", ".join(item.expected_chunk_ids),
                "error": item.error,
            }
        )
    st.dataframe(rows, use_container_width=True)


def _backend_options(default_backends: Sequence[str]) -> List[str]:
    output = ["custom", "ragas", "deepeval"]
    for item in default_backends:
        if item not in output:
            output.append(item)
    return output


def _history() -> List[Dict[str, Any]]:
    import streamlit as st

    if "evaluation_history" not in st.session_state:
        st.session_state["evaluation_history"] = []
    raw = st.session_state["evaluation_history"]
    if not isinstance(raw, list):
        return []
    output: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            output.append(dict(item))
    return output


def _push_history(item: Dict[str, Any]) -> None:
    import streamlit as st

    rows = _history()
    rows.append(dict(item))
    st.session_state["evaluation_history"] = rows[-20:]


def _format_float(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "0.0000"
    return format(float(value), ".4f")
