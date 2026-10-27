import pandas as pd
from src.model import build_pipeline


def test_pipeline_handles_mixed_data():
    x = pd.DataFrame({"income": [10, 20, 30], "job": ["a", "b", "a"]})
    pipeline = build_pipeline(x)
    pipeline.fit(x, [0, 1, 1])
    assert pipeline.predict(x).shape == (3,)
