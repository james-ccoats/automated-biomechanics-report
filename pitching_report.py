import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet

# Load dataset
pitching_poi = pd.read_csv(
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/refs/heads/main/baseball_pitching/data/poi/poi_metrics.csv"
)

# Filter rows
pitching_poi_filtered = pitching_poi[
    (pitching_poi["pitch_speed_mph"] > 88) &
    (pitching_poi["pitch_speed_mph"] < 95) &
    (pitching_poi['p_throws'] == 'R') &
    (pitching_poi['arm_slot'].between(39, 45))
]

# Compute stats
stats = []

for col in pitching_poi_filtered.columns:
    if pd.api.types.is_numeric_dtype(pitching_poi_filtered[col]):
        stats.append({
            "column": col,
            "mean": pitching_poi_filtered[col].mean(),
            "std":  pitching_poi_filtered[col].std()
        })

# Build DataFrame
stats_df = pd.DataFrame(stats).set_index("column").T
print(stats_df)


