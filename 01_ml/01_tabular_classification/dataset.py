from __future__ import annotations

from dataclasses import dataclass

from _runtime import *  # noqa: F401,F403


@dataclass
class AdultClassificationSplit:
    full_frame: pd.DataFrame
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    mlp_preprocessor: Pipeline


def load_adult() -> pd.DataFrame:
    df = load_dataset('scikit-learn/adult-census-income', split='train').to_pandas()
    return df.replace('?', np.nan)


def adult_preprocessors(df: pd.DataFrame) -> tuple[list[str], list[str], ColumnTransformer, Pipeline]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    sparse_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), cat_cols),
        ]
    )
    dense_mlp = Pipeline(
        steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
                    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), cat_cols),
                ]
            )),
            ('scale', MaxAbsScaler()),
        ]
    )
    return num_cols, cat_cols, sparse_preprocessor, dense_mlp


def make_split(seed: int = SEED) -> AdultClassificationSplit:
    full_frame = load_adult()
    X = full_frame.drop(columns=['income'])
    y = (full_frame['income'] == '>50K').astype(int).to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_valid_idx, test_idx = next(sss.split(X, y))
    X_train_valid = X.iloc[train_valid_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train_valid = y[train_valid_idx]
    y_test = y[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=seed)
    train_idx, valid_idx = next(sss2.split(X_train_valid, y_train_valid))
    X_train = X_train_valid.iloc[train_idx].reset_index(drop=True)
    X_valid = X_train_valid.iloc[valid_idx].reset_index(drop=True)
    y_train = y_train_valid[train_idx]
    y_valid = y_train_valid[valid_idx]

    _, _, preprocessor, mlp_preprocessor = adult_preprocessors(X_train)
    return AdultClassificationSplit(
        full_frame=full_frame,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        preprocessor=preprocessor,
        mlp_preprocessor=mlp_preprocessor,
    )
