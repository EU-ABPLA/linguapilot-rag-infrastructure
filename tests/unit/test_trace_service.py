from __future__ import annotations

from observability.dashboard.services.trace_service import TraceService


def test_stage_duration_rows_convert_cumulative_elapsed_to_incremental_elapsed() -> None:
    service = TraceService()
    trace = {
        "stages": [
            {
                "stage": "load",
                "elapsed_ms": 100.0,
                "details": {"method": "document_loader", "provider": "pdf"},
            },
            {
                "stage": "split",
                "elapsed_ms": 140.0,
                "details": {"method": "chunking", "provider": "recursive"},
            },
            {
                "stage": "dense_encoder",
                "elapsed_ms": 400.0,
                "details": {"method": "", "provider": ""},
            },
            {
                "stage": "dense_encoder",
                "elapsed_ms": 650.0,
                "details": {"method": "", "provider": ""},
            },
        ]
    }

    rows = service.stage_duration_rows(trace)

    assert [item["stage_label"] for item in rows] == [
        "load",
        "split",
        "dense_encoder #1",
        "dense_encoder #2",
    ]
    assert [item["duration_ms"] for item in rows] == [100.0, 40.0, 260.0, 250.0]
    assert [item["cumulative_elapsed_ms"] for item in rows] == [100.0, 140.0, 400.0, 650.0]


def test_stage_duration_rows_clamp_negative_elapsed_gaps_to_zero() -> None:
    service = TraceService()
    trace = {
        "stages": [
            {"stage": "load", "elapsed_ms": 100.0, "details": {}},
            {"stage": "split", "elapsed_ms": 90.0, "details": {}},
        ]
    }

    rows = service.stage_duration_rows(trace)

    assert rows[0]["duration_ms"] == 100.0
    assert rows[1]["duration_ms"] == 0.0
