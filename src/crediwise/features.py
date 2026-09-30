"""The feature pipeline.

Every fitted transform lives inside a single ``Pipeline`` so that imputation,
scaling and encoding are fit on training folds only. The original notebook fit
its encoders and imputation on the full dataset before splitting (audit #5).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


def build_preprocessor(include_bureau: bool = True) -> ColumnTransformer:
    numeric = [c for c in cfg.NUMERIC_FEATURES
               if include_bureau or c not in cfg.BUREAU_FEATURES]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("nominal", nominal_pipe, cfg.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
