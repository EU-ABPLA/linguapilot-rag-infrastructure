from __future__ import annotations

from observability.dashboard.services.trace_service import TraceService


def render_page() -> None:
    import streamlit as st

    st.header("Ingestion Traces")
    service = TraceService()
    traces = service.list_traces(trace_type="ingestion")
    if not traces:
        st.info("No ingestion traces found. Run ingestion first.")
        return
    table_rows = []
    for item in traces:
        stages = item.get("stages", [])
        stage_count = len(stages) if isinstance(stages, list) else 0
        table_rows.append(
            {
                "trace_id": str(item.get("trace_id", "")),
                "started_at": str(item.get("started_at", "")),
                "finished_at": str(item.get("finished_at", "")),
                "total_elapsed_ms": item.get("total_elapsed_ms"),
                "stage_count": stage_count,
            }
        )
    st.subheader("Trace History")
    st.dataframe(table_rows, use_container_width=True)
    trace_map = {str(item.get("trace_id", "")): item for item in traces}
    trace_ids = [row["trace_id"] for row in table_rows if row["trace_id"]]
    selected_id = st.selectbox("Select Trace", trace_ids, index=0)
    selected = trace_map.get(selected_id)
    if selected is None:
        return
    st.subheader("Trace Detail")
    st.json(
        {
            "trace_id": selected.get("trace_id"),
            "started_at": selected.get("started_at"),
            "finished_at": selected.get("finished_at"),
            "total_elapsed_ms": selected.get("total_elapsed_ms"),
        }
    )
    stage_rows = service.stage_rows(selected)
    if not stage_rows:
        st.info("No stage data available for this trace.")
        return
    st.subheader("Stage Waterfall")
    _render_stage_chart(stage_rows)
    st.dataframe(stage_rows, use_container_width=True)


def _render_stage_chart(stage_rows):
    import streamlit as st

    try:
        import altair as alt
    except Exception:
        st.bar_chart(stage_rows, x="stage", y="elapsed_ms")
        return
    chart_data = stage_rows
    chart = (
        alt.Chart(alt.Data(values=chart_data))
        .mark_bar()
        .encode(
            x=alt.X("elapsed_ms:Q", title="Elapsed (ms)"),
            y=alt.Y("stage:N", sort=[item["stage"] for item in chart_data]),
            color=alt.Color("stage:N", legend=None),
            tooltip=["stage:N", "elapsed_ms:Q", "method:N", "provider:N"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
