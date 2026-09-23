from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BUILDING_SPLIT_COLUMNS, ENERGY_MIX_COLUMNS
from .validation import validate_building_split, validate_energy_mix

ANOMALY_TYPES = (
    "contextual",
    "correlational",
    "structural",
    "energy_mix",
    "combination",
)

DEFAULT_ANOMALY_DISTRIBUTION = {
    "contextual": 0.20,
    "correlational": 0.25,
    "structural": 0.20,
    "energy_mix": 0.25,
    "combination": 0.10,
}

CONTEXT_COLUMNS = (
    "categorie_activite_majoritaire_efa",
    "sous_categorie_activite_majoritaire_efa",
    "cas_assujettissement_efa",
)
RATIO_COLUMNS = (
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2",
    "ratio_de_consommation_brut_kwh_par_m2",
)
GROUND_TRUTH_COLUMNS = (
    "is_anomaly",
    "anomaly_type",
    "anomaly_severity",
)


def _normalise_percentages(values: pd.Series) -> np.ndarray:
    numeric = values.astype(float).fillna(0).clip(lower=0)
    total = numeric.sum()
    if total <= 0:
        normalised = np.zeros(len(numeric), dtype=float)
        normalised[0] = 100.0
        return normalised
    return (numeric * (100.0 / total)).to_numpy()


def _valid_rows(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = df[columns]
    return (
        values.notna().all(axis=1)
        & values.ge(0).all(axis=1)
        & np.isclose(values.sum(axis=1), 100.0, atol=0.1)
    ).to_numpy()


def _choose_donor(
    rng: np.random.Generator,
    target: int,
    candidates: np.ndarray,
    context_ids: np.ndarray,
    base: pd.DataFrame,
    value_columns: tuple[str, ...],
) -> int:
    target_values = base.loc[target, list(value_columns)].to_numpy(dtype=float)
    candidates = candidates[candidates != target]
    if len(candidates) == 0:
        raise ValueError(f"Could not find a valid donor for row {target}.")
    if context_ids.size:
        different_context = candidates[context_ids[candidates] != context_ids[target]]
        if len(different_context):
            candidates = different_context

    # Most candidate pools contain many valid donors. Sampling avoids scanning
    # the full pool for every injected row.
    for _ in range(64):
        donor = int(rng.choice(candidates))
        donor_values = base.loc[donor, list(value_columns)].to_numpy(dtype=float)
        if not np.isclose(donor_values, target_values, equal_nan=True).all():
            return donor

    # Keep deterministic failure behavior for degenerate datasets where the
    # random attempts did not find a suitable donor.
    for donor in candidates:
        donor_values = base.loc[donor, list(value_columns)].to_numpy(dtype=float)
        if not np.isclose(donor_values, target_values, equal_nan=True).all():
            return int(donor)
    raise ValueError(f"Could not find a valid donor for row {target}.")


def _copy_values(
    result: pd.DataFrame,
    target: int,
    donor: int,
    columns: tuple[str, ...],
) -> None:
    result.loc[target, list(columns)] = result.loc[donor, list(columns)].to_numpy()


def _allocate_counts(total: int, distribution: Mapping[str, float]) -> dict[str, int]:
    names = list(distribution)
    unknown = set(names) - set(ANOMALY_TYPES)
    if unknown:
        raise ValueError(f"Unsupported anomaly types: {sorted(unknown)}")

    weights = np.asarray([distribution[name] for name in names], dtype=float)
    if (
        len(names) == 0
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Anomaly distribution must be non-negative and sum to 1.")

    raw_counts = weights * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - int(counts.sum())
    if remainder:
        for index in np.argsort(-(raw_counts - counts))[:remainder]:
            counts[index] += 1
    return dict(zip(names, counts, strict=True))


def _inject_contextual(
    result: pd.DataFrame,
    base: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    context_ids: np.ndarray,
    contextual_candidates: np.ndarray,
) -> None:
    donor = _choose_donor(
        rng,
        target,
        contextual_candidates,
        context_ids,
        base,
        RATIO_COLUMNS,
    )
    _copy_values(result, target, donor, RATIO_COLUMNS)


def _inject_correlational(
    result: pd.DataFrame,
    base: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    valid_energy: np.ndarray,
    context_ids: np.ndarray,
) -> None:
    donor = _choose_donor(
        rng,
        target,
        np.flatnonzero(valid_energy),
        context_ids,
        base,
        tuple(ENERGY_MIX_COLUMNS),
    )
    result.loc[target, list(ENERGY_MIX_COLUMNS)] = _normalise_percentages(
        base.loc[donor, list(ENERGY_MIX_COLUMNS)]
    )


def _inject_structural(
    result: pd.DataFrame,
    base: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    context_ids: np.ndarray,
    mono_occupation_pools: tuple[np.ndarray, np.ndarray],
) -> None:
    mono_pool, multi_pool = mono_occupation_pools
    donor_pool = (
        multi_pool if bool(base.loc[target, "is_mono_occupation"]) else mono_pool
    )
    donor = _choose_donor(
        rng,
        target,
        donor_pool,
        context_ids,
        base,
        tuple(BUILDING_SPLIT_COLUMNS),
    )
    result.loc[target, list(BUILDING_SPLIT_COLUMNS)] = _normalise_percentages(
        base.loc[donor, list(BUILDING_SPLIT_COLUMNS)]
    )


def _inject_energy_mix(
    result: pd.DataFrame,
    base: pd.DataFrame,
    target: int,
    context_ids: np.ndarray,
    extreme_energy_sources: dict[int, np.ndarray],
) -> None:
    current = base.loc[target, list(ENERGY_MIX_COLUMNS)].to_numpy(dtype=float)
    source_order = extreme_energy_sources.get(
        int(context_ids[target]),
        np.arange(len(ENERGY_MIX_COLUMNS)),
    )
    source = next(
        (
            int(candidate)
            for candidate in source_order
            if not np.isclose(current[candidate], 100.0)
            or not np.isclose(current.sum() - current[candidate], 0.0)
        ),
        int(source_order[0]),
    )
    values = np.zeros(len(ENERGY_MIX_COLUMNS), dtype=float)
    values[source] = 100.0
    result.loc[target, list(ENERGY_MIX_COLUMNS)] = values


def _inject_combination(
    result: pd.DataFrame,
    base: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    context_ids: np.ndarray,
    contextual_candidates: np.ndarray,
    mono_occupation_pools: tuple[np.ndarray, np.ndarray],
    extreme_energy_sources: dict[int, np.ndarray],
) -> None:
    _inject_contextual(
        result,
        base,
        target,
        rng,
        context_ids,
        contextual_candidates,
    )
    _inject_energy_mix(
        result=result,
        base=base,
        target=target,
        context_ids=context_ids,
        extreme_energy_sources=extreme_energy_sources,
    )
    _inject_structural(
        result,
        base,
        target,
        rng,
        context_ids,
        mono_occupation_pools,
    )


def _recalculate_derived_columns(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["sum_energy_mix_percent"] = result[ENERGY_MIX_COLUMNS].sum(axis=1)
    result["is_valid_energy_mix"] = validate_energy_mix(result)
    result["sum_building_split_percent"] = result[BUILDING_SPLIT_COLUMNS].sum(axis=1)
    result["is_valid_building_split"] = validate_building_split(result)
    result["delta_climat_kwh_m2"] = result[RATIO_COLUMNS[0]] - result[RATIO_COLUMNS[1]]
    return result


def validate_injected_dataset(
    experiment_df: pd.DataFrame,
    clean_df: pd.DataFrame | None = None,
) -> None:
    required = set(GROUND_TRUTH_COLUMNS) | {
        "source_row_id",
        *ENERGY_MIX_COLUMNS,
        *BUILDING_SPLIT_COLUMNS,
        *RATIO_COLUMNS,
        "sum_energy_mix_percent",
        "is_valid_energy_mix",
        "sum_building_split_percent",
        "is_valid_building_split",
        "delta_climat_kwh_m2",
    }
    missing = required - set(experiment_df.columns)
    if missing:
        raise ValueError(f"Experiment dataset is missing columns: {sorted(missing)}")
    if (
        experiment_df["source_row_id"].isna().any()
        or not experiment_df["source_row_id"].is_unique
    ):
        raise ValueError("source_row_id must be present and unique.")

    anomalies = experiment_df["is_anomaly"].astype(bool)
    if not experiment_df.loc[anomalies, "anomaly_type"].isin(ANOMALY_TYPES).all():
        raise ValueError("Every anomaly must have a supported anomaly_type.")
    if not (experiment_df.loc[~anomalies, "anomaly_type"] == "none").all():
        raise ValueError("Clean rows must have anomaly_type='none'.")
    if not experiment_df.loc[anomalies, "is_valid_energy_mix"].all():
        raise ValueError("Injected anomalies must preserve energy-mix validity.")
    if not experiment_df.loc[anomalies, "is_valid_building_split"].all():
        raise ValueError("Injected anomalies must preserve building-split validity.")

    expected = _recalculate_derived_columns(experiment_df)
    for column in (
        "sum_energy_mix_percent",
        "is_valid_energy_mix",
        "sum_building_split_percent",
        "is_valid_building_split",
        "delta_climat_kwh_m2",
    ):
        if not experiment_df[column].equals(expected[column]):
            raise ValueError(f"Derived column is inconsistent: {column}")

    if clean_df is not None:
        if set(experiment_df["source_row_id"]) != set(clean_df["source_row_id"]):
            raise ValueError("Experiment provenance does not match the clean dataset.")
        clean_by_source = clean_df.set_index("source_row_id")
        comparable = [
            column
            for column in clean_df.columns
            if column not in {*GROUND_TRUTH_COLUMNS, "source_row_id"}
        ]
        mutated = experiment_df.loc[anomalies].set_index("source_row_id")[comparable]
        original = clean_by_source.loc[mutated.index, comparable]
        differences = mutated.reset_index(drop=True).compare(
            original.reset_index(drop=True)
        )
        changed_rows = differences.index.unique()
        if len(changed_rows) != len(mutated):
            raise ValueError("Every anomaly must differ from its clean source row.")


def inject_anomalies(
    clean_df: pd.DataFrame,
    anomaly_rate: float = 0.05,
    random_seed: int = 42,
    anomaly_type_distribution: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if not 0 <= anomaly_rate <= 1:
        raise ValueError("anomaly_rate must be between 0 and 1.")
    if len(clean_df) < 2 and anomaly_rate:
        raise ValueError("At least two rows are required to inject anomalies.")
    if "source_row_id" not in clean_df.columns:
        clean_df = clean_df.copy()
        clean_df.insert(0, "source_row_id", pd.Series(clean_df.index, dtype="int64"))

    required = (
        set(ENERGY_MIX_COLUMNS) | set(BUILDING_SPLIT_COLUMNS) | set(RATIO_COLUMNS)
    )
    required.add("is_mono_occupation")
    missing = required - set(clean_df.columns)
    if missing:
        raise ValueError(f"Clean dataset is missing columns: {sorted(missing)}")

    base = clean_df.copy(deep=True)
    result = base.copy(deep=True)
    result["is_anomaly"] = False
    result["anomaly_type"] = "none"
    result["anomaly_severity"] = "none"

    anomaly_count = round(len(result) * anomaly_rate)
    distribution = anomaly_type_distribution or DEFAULT_ANOMALY_DISTRIBUTION
    counts = _allocate_counts(anomaly_count, distribution)
    rng = np.random.default_rng(random_seed)
    targets = rng.choice(len(result), size=anomaly_count, replace=False)

    valid_energy = _valid_rows(base, list(ENERGY_MIX_COLUMNS))
    valid_splits = _valid_rows(base, list(BUILDING_SPLIT_COLUMNS))
    valid_ratios = base[list(RATIO_COLUMNS)].notna().all(axis=1).to_numpy()
    context_ids = pd.factorize(
        pd.MultiIndex.from_frame(base[list(CONTEXT_COLUMNS)].astype("string"))
    )[0]
    ratio_values = base[RATIO_COLUMNS[0]].to_numpy(dtype=float, na_value=np.nan)
    valid_ratio_indices = np.flatnonzero(valid_ratios)
    ordered_ratios = valid_ratio_indices[
        np.argsort(ratio_values[valid_ratio_indices], kind="stable")
    ]
    tail_size = max(1, len(ordered_ratios) // 10)
    contextual_candidates = np.unique(
        np.concatenate((ordered_ratios[:tail_size], ordered_ratios[-tail_size:]))
    )
    mono_values = base["is_mono_occupation"].to_numpy(dtype=bool)
    mono_occupation_pools = (
        np.flatnonzero(valid_splits & ~mono_values),
        np.flatnonzero(valid_splits & mono_values),
    )
    energy_values = base[list(ENERGY_MIX_COLUMNS)].to_numpy(dtype=float)
    global_energy_mean = np.nanmean(energy_values[valid_energy], axis=0)
    extreme_energy_sources: dict[int, np.ndarray] = {}
    for context_id in np.unique(context_ids):
        context_rows = valid_energy & (context_ids == context_id)
        context_mean = (
            np.nanmean(energy_values[context_rows], axis=0)
            if context_rows.any()
            else global_energy_mean
        )
        dominant_source = int(np.nanargmax(context_mean))
        source_order = np.argsort(context_mean, kind="stable")
        source_order = source_order[source_order != dominant_source]
        extreme_energy_sources[int(context_id)] = np.concatenate(
            (source_order, np.array([dominant_source]))
        )
    injectors = {
        "contextual": _inject_contextual,
        "correlational": _inject_correlational,
        "structural": _inject_structural,
        "energy_mix": _inject_energy_mix,
        "combination": _inject_combination,
    }
    severity = {
        "contextual": "high",
        "correlational": "medium",
        "structural": "medium",
        "energy_mix": "high",
        "combination": "high",
    }

    offset = 0
    for anomaly_type, count in counts.items():
        injector = injectors[anomaly_type]
        for target in targets[offset : offset + count]:
            target = int(target)
            if anomaly_type in {"contextual"}:
                injector(
                    result,
                    base,
                    target,
                    rng,
                    context_ids,
                    contextual_candidates,
                )
            elif anomaly_type in {"correlational", "energy_mix"}:
                if anomaly_type == "energy_mix":
                    injector(
                        result,
                        base,
                        target,
                        context_ids,
                        extreme_energy_sources,
                    )
                else:
                    injector(result, base, target, rng, valid_energy, context_ids)
            elif anomaly_type == "structural":
                injector(
                    result,
                    base,
                    target,
                    rng,
                    context_ids,
                    mono_occupation_pools,
                )
            else:
                injector(
                    result,
                    base,
                    target,
                    rng,
                    context_ids,
                    contextual_candidates,
                    mono_occupation_pools,
                    extreme_energy_sources,
                )
            result.loc[target, "is_anomaly"] = True
            result.loc[target, "anomaly_type"] = anomaly_type
            result.loc[target, "anomaly_severity"] = severity[anomaly_type]
        offset += count

    result = _recalculate_derived_columns(result).reset_index(drop=True)
    validate_injected_dataset(result, base)
    return result


def create_experiment_dataset(
    clean_path: str | Path,
    experiment_path: str | Path,
    anomaly_rate: float = 0.05,
    random_seed: int = 42,
    anomaly_type_distribution: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    clean_df = pd.read_parquet(clean_path)
    experiment_df = inject_anomalies(
        clean_df,
        anomaly_rate=anomaly_rate,
        random_seed=random_seed,
        anomaly_type_distribution=anomaly_type_distribution,
    )
    output_path = Path(experiment_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_df.to_parquet(output_path, index=False)
    return experiment_df
