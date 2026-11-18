from pathlib import Path

import pytest

from app.logic import SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=["baseline","circuit","no_fcy"])
def test_page_load_and_single_car_evaluation(app_inputs, scenario):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    page = AppTest.from_file(str(root / "app/strategy_app.py"), default_timeout=120).run()
    assert not page.exception and not page.error
    assert page.title[0].value == "F1 策略模拟器"
    next(box for box in page.selectbox if box.label == "情景").set_value(scenario)
    page.button[0].click().run()
    assert not page.exception and not page.error
    results = page.dataframe[-1].value
    assert len(results) == 3
    assert {"平均秒数", "CVaR95", "配对均差", "最快占比"} <= set(results.columns)
    assert results["平均秒数"].is_monotonic_increasing
    assert results["配对均差"].iloc[0] == 0
    base = next(box for box in page.selectbox if box.label == "基准策略")
    assert base.value == results["策略"].iloc[0]
    base.set_value(results["策略"].iloc[-1]).run()
    assert not page.exception
    changed = page.dataframe[-1].value
    assert changed["配对均差"].iloc[-1] == 0
    assert changed["配对均差"].iloc[0] < 0
    assert "95% 配对差区间" in page.info[-1].value


def test_multicar_result_for_selected_driver(app_inputs):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    page = AppTest.from_file(str(root / "app/strategy_app.py"), default_timeout=120).run()
    page.toggle[0].set_value(True).run()
    page.slider[0].set_value(500)
    page.button[0].click().run()
    assert not page.exception and not page.error
    result = page.dataframe[-1].value
    assert len(result) == 3
    assert result["期望名次"].ge(1).all()
    assert "提升（位）" in page.dataframe[-2].value.columns
    assert page.get("arrow_vega_lite_chart") or page.get("vega_lite_chart")


def test_prerace_evaluation_and_unavailable_season(app_inputs, prior_inputs):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    page = AppTest.from_file(str(root / "app/strategy_app.py"), default_timeout=120).run()
    assert page.radio[0].value == "事后"
    page.radio[0].set_value("赛前").run()
    page.slider[0].set_value(500)
    page.button[0].click().run()
    assert not page.exception and not page.error
    assert set(page.dataframe[-1].value["策略"]) == {"最优一停", "最优两停"}
    page.toggle[0].set_value(True).run()
    page.button[0].click().run()
    assert not page.exception and not page.error
    assert "提升（位）" in page.dataframe[-2].value.columns
    assert any("相对赛前单车最优策略" in item.value for item in page.markdown)
    page.toggle[0].set_value(False).run()
    next(box for box in page.selectbox if box.label == "分站").set_value("2024-5").run()
    assert not page.exception and "no_base_lap" in page.warning[0].value
    next(box for box in page.selectbox if box.label == "赛季").set_value(2023).run()
    assert not page.exception and "no_prior_season" in page.warning[0].value


def test_clean_checkout_mode_selector():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    if (root / "data/processed/model_laps.parquet").exists():
        pytest.skip("Requires an empty data directory")
    page = AppTest.from_file(str(root / "app/strategy_app.py"), default_timeout=120).run()
    assert page.radio[0].value == "事后"
    page.radio[0].set_value("赛前").run()
    assert not page.exception and not page.error
    assert "scripts/download_data.md" in page.info[-1].value
