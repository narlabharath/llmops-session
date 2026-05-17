from __future__ import annotations

import importlib


def test_streamlit_app_scaffold_imports() -> None:
    module = importlib.import_module("src.app.streamlit_app")

    assert module.PAGE_TITLE == "Learning Program Support Assistant"
    assert module.DEFAULT_PROVIDER == "anthropic"
    assert callable(module.render_question_panel)
    assert callable(module.render_answer_panel)
    assert callable(module.render_trace_panel)
    assert callable(module.render_history_panel)
