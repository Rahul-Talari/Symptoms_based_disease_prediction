from symptom_dx.cli import main


def test_predict_without_api_key_prints_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert main(["predict", "fever", "--sources", "web"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_plot_creates_both_charts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.csv").write_text(
        "metric,complexity,technique,value\nBLEU-1,Simple,1,0.9\nBLEU-1,Complex,1,0.8\n"
    )
    (tmp_path / "results" / "inference_time.csv").write_text("complexity,technique,value\nSimple,1,20.5\n")

    assert main(["plot"]) == 0
    assert (tmp_path / "results" / "figures" / "metrics_plot.png").exists()
    assert (tmp_path / "results" / "figures" / "inference_plot.png").exists()
