import pandas as pd
from fedids_bench.reporting.tables import generate_latex_table

def test_latex_table_generator():
    df = pd.DataFrame({
        "Algorithm": ["FedAvg", "FedProx", "FedAdam"],
        "F1": [0.98, 0.97, 0.985],
        "Bytes": [10240, 10240, 10240]
    })
    latex = generate_latex_table(df)
    assert "\\begin{table}" in latex
    assert "FedAvg" in latex
    assert "0.9800" in latex
