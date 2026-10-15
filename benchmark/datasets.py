import os

import pandas as pd

from benchmark.serialization import SERIALIZATION_MODES, serialize_record

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


DATASET_SPECS = {
    "abt-buy": {
        "directory": "abt-buy",
        "entity_type": "product records",
        "source_a": {
            "filename": "Abt.csv",
            "source": "abt",
            "ground_truth_aliases": ("abt", "idabt"),
        },
        "source_b": {
            "filename": "Buy.csv",
            "source": "buy",
            "ground_truth_aliases": ("buy", "idbuy"),
        },
        "ground_truth_filename": "abt_buy_perfectMapping.csv",
    },
    "dblp-acm": {
        "directory": "dblp-acm",
        "entity_type": "publication records",
        "source_a": {
            "filename": "DBLP2.csv",
            "source": "dblp",
            "ground_truth_aliases": ("dblp", "iddblp"),
        },
        "source_b": {
            "filename": "ACM.csv",
            "source": "acm",
            "ground_truth_aliases": ("acm", "idacm"),
        },
        "ground_truth_filename": "DBLP-ACM_perfectMapping.csv",
    },
    "dblp-scholar": {
        "directory": "dblp-scholar",
        "entity_type": "publication records",
        "source_a": {
            "filename": "DBLP1.csv",
            "source": "dblp",
            "ground_truth_aliases": ("dblp", "iddblp"),
        },
        "source_b": {
            "filename": "Scholar.csv",
            "source": "scholar",
            "ground_truth_aliases": ("scholar", "idscholar"),
        },
        "ground_truth_filename": "DBLP-Scholar_perfectMapping.csv",
    },
    "amazon-google": {
        "directory": "amazon-google",
        "entity_type": "product records",
        "source_a": {
            "filename": "Amazon.csv",
            "source": "amazon",
            "ground_truth_aliases": ("amazon", "idamazon"),
        },
        "source_b": {
            "filename": "GoogleProducts.csv",
            "source": "google",
            "ground_truth_aliases": ("google", "idgoogle"),
        },
        "ground_truth_filename": "Amzon_GoogleProducts_perfectMapping.csv",
    },
    "wdc-block-small": {
        "directory": "wdcproducts80cc20rnd050un_block_s_train_s",
        "entity_type": "product offers",
        "source_a": {
            "filename": "tableA.csv",
            "source": "wdc_a",
            "ground_truth_aliases": ("ltable_id",),
            "exclude_columns": ("cluster_id", "page_url"),
        },
        "source_b": {
            "filename": "tableB.csv",
            "source": "wdc_b",
            "ground_truth_aliases": ("rtable_id",),
            "exclude_columns": ("cluster_id", "page_url"),
        },
        "ground_truth_filename": "test.csv",
        "ground_truth_label_column": "label",
    },
}


def _read_csv(path, **kwargs):
    """Try multiple encodings to read a CSV."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path} with any encoding")


def _normalize_source_frame(
    df,
    source_name,
    serialization_mode="values_space",
    exclude_columns=(),
):
    id_col = [c for c in df.columns if "id" in c.lower()][0]
    normalized = df.rename(columns={id_col: "id"}).copy()
    normalized["id"] = normalized["id"].astype(str)
    normalized["source"] = source_name

    excluded = {column.lower() for column in exclude_columns}
    text_cols = [
        c for c in normalized.columns
        if c not in ("id", "source")
        and c.lower() not in excluded
        and not c.startswith("Unnamed")
    ]

    records = {}
    for row in normalized[["id"] + text_cols].itertuples(index=False):
        row_id = str(row[0])
        fields = {
            col: ("" if pd.isna(value) else str(value))
            for col, value in zip(text_cols, row[1:])
        }
        records[row_id] = fields

    if serialization_mode not in SERIALIZATION_MODES:
        raise ValueError(f"Unknown serialization mode: {serialization_mode}")

    normalized["text"] = normalized["id"].map(
        lambda row_id: serialize_record(records[str(row_id)], serialization_mode)
    )

    return normalized, text_cols, records


def _ground_truth_columns(gt_df, source_a_spec, source_b_spec):
    gt_columns = [c.lower() for c in gt_df.columns]

    def resolve_column(aliases):
        for alias in aliases:
            for index, column in enumerate(gt_columns):
                if alias in column:
                    return gt_df.columns[index]
        raise ValueError(f"Could not find ground truth column for aliases {aliases!r}")

    col_a = resolve_column(source_a_spec["ground_truth_aliases"])
    col_b = resolve_column(source_b_spec["ground_truth_aliases"])
    return col_a, col_b


def _build_dataset_bundle(name, serialization_mode="values_space"):
    spec = DATASET_SPECS[name]
    base = os.path.join(DATA_DIR, spec["directory"])

    df_a = _read_csv(os.path.join(base, spec["source_a"]["filename"]))
    df_b = _read_csv(os.path.join(base, spec["source_b"]["filename"]))
    gt = _read_csv(os.path.join(base, spec["ground_truth_filename"]))

    frame_a, cols_a, records_a = _normalize_source_frame(
        df_a,
        spec["source_a"]["source"],
        serialization_mode=serialization_mode,
        exclude_columns=spec["source_a"].get("exclude_columns", ()),
    )
    frame_b, cols_b, records_b = _normalize_source_frame(
        df_b,
        spec["source_b"]["source"],
        serialization_mode=serialization_mode,
        exclude_columns=spec["source_b"].get("exclude_columns", ()),
    )

    combined = pd.concat(
        [
            frame_a[["id", "source", "text"]],
            frame_b[["id", "source", "text"]],
        ],
        ignore_index=True,
    )

    gt_col_a, gt_col_b = _ground_truth_columns(gt, spec["source_a"], spec["source_b"])
    labeled_pairs = None
    label_column = spec.get("ground_truth_label_column")
    if label_column:
        labeled_pairs = {
            (str(a), str(b))
            for a, b in zip(gt[gt_col_a], gt[gt_col_b])
        }
        positive_rows = gt[gt[label_column] == 1]
    else:
        positive_rows = gt

    ground_truth = {
        (str(a), str(b))
        for a, b in zip(positive_rows[gt_col_a], positive_rows[gt_col_b])
    }

    return {
        "combined": combined,
        "ground_truth": ground_truth,
        "labeled_pairs": labeled_pairs,
        "entity_type": spec["entity_type"],
        "source_a": spec["source_a"]["source"],
        "source_b": spec["source_b"]["source"],
        "serialization_mode": serialization_mode,
        "records": {
            spec["source_a"]["source"]: records_a,
            spec["source_b"]["source"]: records_b,
        },
        "columns": {
            spec["source_a"]["source"]: cols_a,
            spec["source_b"]["source"]: cols_b,
        },
    }


def load_dataset(name, serialization_mode="values_space"):
    """Returns (combined_df, ground_truth_set, (source_a, source_b))."""
    bundle = _build_dataset_bundle(name, serialization_mode=serialization_mode)
    return bundle["combined"], bundle["ground_truth"], (bundle["source_a"], bundle["source_b"])


def load_dataset_with_evaluation(name, serialization_mode="values_space"):
    """Load a dataset plus an optional labeled evaluation-pair universe."""
    bundle = _build_dataset_bundle(name, serialization_mode=serialization_mode)
    return (
        bundle["combined"],
        bundle["ground_truth"],
        (bundle["source_a"], bundle["source_b"]),
        bundle["labeled_pairs"],
    )


def load_processing_dataset(name):
    """Return structured dataset metadata for the LLM processing benchmark."""
    bundle = _build_dataset_bundle(name)
    return {
        "name": name,
        "entity_type": bundle["entity_type"],
        "combined_df": bundle["combined"],
        "ground_truth": bundle["ground_truth"],
        "labeled_pairs": bundle["labeled_pairs"],
        "source_a": bundle["source_a"],
        "source_b": bundle["source_b"],
        "records": bundle["records"],
        "columns": bundle["columns"],
        "n_records_a": len(bundle["records"][bundle["source_a"]]),
        "n_records_b": len(bundle["records"][bundle["source_b"]]),
    }


DATASETS = {
    name: (lambda dataset_name=name: load_dataset(dataset_name))
    for name in DATASET_SPECS
}
