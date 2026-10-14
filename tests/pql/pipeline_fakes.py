"""Deterministic local Nori for the pipeline round-trip tests.

The test process and the child process both install these fakes, so a
prediction depends only on the inputs Nori receives.
"""

import hashlib
import os

import numpy as np

#: The child process and the test agree on this fake Hugging Face revision.
FAKE_REVISION = "0123456789abcdef0123456789abcdef01234567"
INPUTS_LOG_ENV = "NORI_REL_TEST_PIPELINE_INPUTS_LOG"


class InputHashRegressor:
    """Predict from the context and query rows, and log a hash of them."""

    def __init__(self, **options) -> None:
        self.options = options
        self.categorical_columns = "auto"

    def get_params(self):
        return {
            "augmentations": ("yj",),
            "categorical_columns": self.categorical_columns,
        }

    def set_params(self, **params):
        self.categorical_columns = params["categorical_columns"]

    def fit(self, train, target):
        self.train, self.target = train, target
        return self

    def predict(self, query, *, output_type="mean", quantiles=None):
        digest = hashlib.sha256()
        for frame in (self.train, self.target, query):
            digest.update(frame.to_csv().encode())
        digest.update(repr(self.categorical_columns).encode())
        with open(os.environ[INPUTS_LOG_ENV], "a", encoding="utf-8") as log:
            log.write(digest.hexdigest() + "\n")
        numeric = query.select_dtypes("number").astype(float).fillna(0.0)
        return (
            float(self.target.mean())
            + 1e-3 * numeric.sum(axis=1).to_numpy(dtype=float)
            + np.zeros(len(query))
        )


def install(checkpoint_path: str) -> None:
    """Replace the regressor and the checkpoint download in this process."""
    import synthefy_nori

    import nori_rel._checkpoint

    synthefy_nori.NoriRegressor = InputHashRegressor
    nori_rel._checkpoint.get_checkpoint_path = lambda model, revision: str(
        checkpoint_path
    )


def last_inputs_hash() -> str:
    with open(os.environ[INPUTS_LOG_ENV], encoding="utf-8") as log:
        return log.read().split()[-1]
