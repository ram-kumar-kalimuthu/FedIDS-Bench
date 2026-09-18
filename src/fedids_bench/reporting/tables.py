import pandas as pd

def generate_latex_table(df: pd.DataFrame, caption: str = "Benchmark Summary Results", label: str = "tab:benchmark_results") -> str:
    """
    Generate publication-ready LaTeX table code from a summary DataFrame.
    """
    latex_str = df.to_latex(
        index=False,
        float_format="{:.4f}".format,
        column_format="l" * len(df.columns),
        caption=caption,
        label=label,
        escape=False
    )
    return latex_str
