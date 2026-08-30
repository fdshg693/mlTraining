"""Download the original handson-ml3 README and root-level notebooks.

Fetches the upstream README.md and every root-level ``*.ipynb`` file from
https://github.com/ageron/handson-ml3 and places them under
``handson-ml3/original/``. The notebook list is fetched dynamically from the
GitHub API; if that call fails (e.g. no network, rate limit), a fixed
fallback list is used instead.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

GITHUB_OWNER = "ageron"
GITHUB_REPO = "handson-ml3"
BRANCH = "main"

CONTENTS_API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/"
)
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{BRANCH}"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "handson-ml3" / "original"

# Fallback used when the GitHub API listing cannot be fetched.
FALLBACK_NOTEBOOKS = [
    "01_the_machine_learning_landscape.ipynb",
    "02_end_to_end_machine_learning_project.ipynb",
    "03_classification.ipynb",
    "04_training_linear_models.ipynb",
    "05_support_vector_machines.ipynb",
    "06_decision_trees.ipynb",
    "07_ensemble_learning_and_random_forests.ipynb",
    "08_dimensionality_reduction.ipynb",
    "09_unsupervised_learning.ipynb",
    "10_neural_nets_with_keras.ipynb",
    "11_training_deep_neural_networks.ipynb",
    "12_custom_models_and_training_with_tensorflow.ipynb",
    "13_loading_and_preprocessing_data.ipynb",
    "14_deep_computer_vision_with_cnns.ipynb",
    "15_processing_sequences_using_rnns_and_cnns.ipynb",
    "16_nlp_with_rnns_and_attention.ipynb",
    "17_autoencoders_gans_and_diffusion_models.ipynb",
    "18_reinforcement_learning.ipynb",
    "19_training_and_deploying_at_scale.ipynb",
    "extra_ann_architectures.ipynb",
    "extra_autodiff.ipynb",
    "extra_gradient_descent_comparison.ipynb",
    "index.ipynb",
    "math_differential_calculus.ipynb",
    "math_linear_algebra.ipynb",
    "tools_matplotlib.ipynb",
    "tools_numpy.ipynb",
    "tools_pandas.ipynb",
]


def list_root_notebooks() -> list[str]:
    """Return the root-level ``*.ipynb`` filenames, fetched from the GitHub API.

    Falls back to :data:`FALLBACK_NOTEBOOKS` if the API call fails.
    """

    try:
        with urllib.request.urlopen(CONTENTS_API_URL, timeout=10) as response:
            entries = json.load(response)
        notebooks = sorted(
            entry["name"]
            for entry in entries
            if entry.get("type") == "file" and entry["name"].endswith(".ipynb")
        )
        if notebooks:
            return notebooks
    except Exception as error:  # noqa: BLE001 - any failure means "use fallback"
        print(f"could not list notebooks via GitHub API ({error}); using fallback list")
    return list(FALLBACK_NOTEBOOKS)


def download(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"downloaded {url} -> {dest_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download(f"{RAW_BASE_URL}/README.md", OUTPUT_DIR / "README.md")

    for notebook_name in list_root_notebooks():
        download(f"{RAW_BASE_URL}/{notebook_name}", OUTPUT_DIR / notebook_name)


if __name__ == "__main__":
    main()
