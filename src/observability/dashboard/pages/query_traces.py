from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from observability.dashboard.services.trace_service import TraceService


def render_page() -> None:
    import streamlit as st

    st.header("Query Traces")
    service = TraceService()
    traces = service.list_traces(trace_type="query")
    if not traces:
        st.info("No query traces found. Run query first.")
        return
    keyword = st.text_input("Search Query Keyword", value="")
    filtered = _filter_by_keyword(traces, keyword)
    if not filtered:
        st.info("No query traces match current keyword.")
        return
    history_rows: List[Dict[str, Any]] = []
    for item in filtered:
        stages = item.get("stages", [])
        stage_count = len(stages) if isinstance(stages, list) else 0
        history_rows.append(
            {
                "trace_id": str(item.get("trace_id", "")),
                "query": _extract_query_text(item),
                "started_at": str(item.get("started_at", "")),
                "finished_at": str(item.get("finished_at", "")),
                "total_elapsed_ms": item.get("total_elapsed_ms"),
                "stage_count": stage_count,
                "error": str(item.get("error", "")),
            }
        )
    st.subheader("Trace History")
    st.dataframe(history_rows, use_container_width=True)
    trace_map = {str(item.get("trace_id", "")): item for item in filtered}
    trace_ids = [row["trace_id"] for row in history_rows if row["trace_id"]]
    selected_id = st.selectbox("Select Trace", trace_ids, index=0)
    selected = trace_map.get(selected_id)
    if selected is None:
        return
    st.subheader("Trace Detail")
    st.json(
        {
            "trace_id": selected.get("trace_id"),
            "query": _extract_query_text(selected),
            "started_at": selected.get("started_at"),
            "finished_at": selected.get("finished_at"),
            "total_elapsed_ms": selected.get("total_elapsed_ms"),
            "error": selected.get("error"),
        }
    )
    stage_rows = service.stage_rows(selected)
    if stage_rows:
        st.subheader("Stage Waterfall")
        _render_stage_chart(stage_rows)
        st.dataframe(stage_rows, use_container_width=True)
    else:
        st.info("No stage data available for this trace.")
    st.subheader("Dense vs Sparse")
    dense = _stage_summary(selected, "dense_retrieval")
    sparse = _stage_summary(selected, "sparse_retrieval")
    col_dense, col_sparse = st.columns(2)
    with col_dense:
        st.markdown("**Dense Retrieval**")
        st.json(dense)
    with col_sparse:
        st.markdown("**Sparse Retrieval**")
        st.json(sparse)
    st.dataframe(_dense_sparse_compare_rows(dense, sparse), use_container_width=True)
    st.subheader("Rerank Change")
    rerank = _stage_summary(selected, "rerank")
    if not rerank:
        rerank = _stage_summary(selected, "reranker")
    st.json(rerank)
    before_ids = _extract_rank_ids(selected, before=True)
    after_ids = _extract_rank_ids(selected, before=False)
    if before_ids and after_ids:
        st.dataframe(_build_rank_change_rows(before_ids, after_ids), use_container_width=True)
    else:
        st.info("No rank-list payload found in trace. Showing rerank stage summary only.")


def _render_stage_chart(stage_rows: Sequence[Mapping[str, Any]]) -> None:
    import streamlit as st

    chart_rows = [dict(item) for item in stage_rows]
    try:
        import altair as alt
    except Exception:
        st.bar_chart(chart_rows, x="stage", y="elapsed_ms")
        return
    order = [str(item.get("stage", "")) for item in chart_rows]
    chart = (
        alt.Chart(alt.Data(values=chart_rows))
        .mark_bar()
        .encode(
            x=alt.X("elapsed_ms:Q", title="Elapsed (ms)"),
            y=alt.Y("stage:N", sort=order),
            color=alt.Color("stage:N", legend=None),
            tooltip=["stage:N", "elapsed_ms:Q", "method:N", "provider:N"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _filter_by_keyword(
    traces: Sequence[Mapping[str, Any]],
    keyword: str,
) -> List[Dict[str, Any]]:
    normalized = keyword.strip().lower()
    if not normalized:
        return [dict(item) for item in traces]
    output: List[Dict[str, Any]] = []
    for item in traces:
        haystack = " ".join(
            [
                str(item.get("trace_id", "")),
                _extract_query_text(item),
                _safe_json(item.get("metadata")),
                _safe_json(item.get("stages")),
            ]
        ).lower()
        if normalized in haystack:
            output.append(dict(item))
    return output


def _extract_query_text(trace: Mapping[str, Any]) -> str:
    metadata = trace.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("query", "query_text", "user_query", "question", "input_query"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for stage in _stages(trace):
        details = stage.get("details")
        if not isinstance(details, Mapping):
            continue
        for key in ("query", "query_text", "question"):
            value = details.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _stage_summary(trace: Mapping[str, Any], stage_name: str) -> Dict[str, Any]:
    entries = [item for item in _stages(trace) if item.get("stage") == stage_name]
    if not entries:
        return {}
    selected = entries[-1]
    details = selected.get("details")
    payload = dict(details) if isinstance(details, Mapping) else {}
    payload["stage"] = stage_name
    payload["elapsed_ms"] = selected.get("elapsed_ms")
    payload["timestamp"] = selected.get("timestamp")
    return payload


def _dense_sparse_compare_rows(
    dense: Mapping[str, Any],
    sparse: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in (
        "status",
        "method",
        "provider",
        "elapsed_ms",
        "top_k",
        "keyword_count",
        "result_count",
        "fallback",
        "reason",
    ):
        rows.append(
            {
                "metric": key,
                "dense": dense.get(key),
                "sparse": sparse.get(key),
            }
        )
    return rows


def _extract_rank_ids(trace: Mapping[str, Any], before: bool) -> List[str]:
    metadata = trace.get("metadata")
    if isinstance(metadata, Mapping):
        keys = (
            ("fusion_ranked_ids", "before_rerank_ids", "hybrid_ranked_ids")
            if before
            else ("reranked_ids", "after_rerank_ids", "final_ranked_ids")
        )
        for key in keys:
            value = metadata.get(key)
            parsed = _normalize_id_list(value)
            if parsed:
                return parsed
    if before:
        fusion = _stage_summary(trace, "fusion")
        parsed = _normalize_id_list(
            fusion.get("ranked_ids")
            or fusion.get("result_ids")
            or fusion.get("chunk_ids")
        )
        if parsed:
            return parsed
    rerank = _stage_summary(trace, "rerank")
    parsed = _normalize_id_list(
        rerank.get("ranked_ids")
        or rerank.get("result_ids")
        or rerank.get("chunk_ids")
    )
    if parsed:
        return parsed
    return []


def _build_rank_change_rows(
    before_ids: Sequence[str],
    after_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    before_index: Dict[str, int] = {}
    for index, chunk_id in enumerate(before_ids, start=1):
        if chunk_id and chunk_id not in before_index:
            before_index[chunk_id] = index
    rows: List[Dict[str, Any]] = []
    for index, chunk_id in enumerate(after_ids, start=1):
        original = before_index.get(chunk_id)
        delta: Optional[int]
        if original is None:
            delta = None
        else:
            delta = original - index
        rows.append(
            {
                "chunk_id": chunk_id,
                "before_rank": original,
                "after_rank": index,
                "rank_delta": delta,
            }
        )
    return rows


def _normalize_id_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
    return output


def _stages(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stages = trace.get("stages")
    if not isinstance(stages, list):
        return []
    output: List[Dict[str, Any]] = []
    for item in stages:
        if isinstance(item, Mapping):
            output.append(dict(item))
    return output


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
