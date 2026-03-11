"""
2x4 panel plots (2 rows AU/EU x 4 models) for all four OOD conditions.
Generates both variance (mean +/- std bands) and entropy (line plots) figures.
One variance + one entropy figure per condition: linear homo, linear hetero, sin homo, sin hetero.
Loads npz from results/ood/outputs/ood/<noise_type>/<func_type>/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_roo