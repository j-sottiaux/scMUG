"""Strict loader for dataset-specific DMKCN lambda triplets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import yaml


EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign_id",
    "fallback",
    "candidates",
    "datasets",
}
LAMBDA_KEYS = ("lambda1", "lambda2", "lambda3")


@dataclass(frozen=True)
class LambdaTriplet:
    """Resolved, validated lambda configuration for one dataset."""

    dataset: str
    lambda_config_id: str
    lambda1: float
    lambda2: float
    lambda3: float
    campaign_id: str
    source_file: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LambdaConfiguration:
    """Validated campaign-wide lambda configuration."""

    campaign_id: str
    candidates: dict[str, tuple[float, float, float]]
    datasets: dict[str, str]
    fallback: str
    source_file: str

    def resolve_candidate(self, dataset: str, candidate_id: str) -> LambdaTriplet:
        """Resolve an explicitly selected, validated candidate for one dataset.

        This is used by experiments that reuse the frozen candidate registry
        without changing the dataset mapping of the campaign that created it.
        """
        normalized = str(dataset).strip().casefold()
        selected = str(candidate_id).strip()
        if not normalized:
            raise ValueError("dataset must be a non-empty string.")
        if selected not in self.candidates:
            raise KeyError(
                f"Unknown lambda candidate {candidate_id!r} in {self.source_file}; "
                f"available candidates are {sorted(self.candidates)}."
            )
        lambda1, lambda2, lambda3 = self.candidates[selected]
        return LambdaTriplet(
            dataset=normalized,
            lambda_config_id=selected,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            campaign_id=self.campaign_id,
            source_file=self.source_file,
        )

    def resolve(self, dataset: str) -> LambdaTriplet:
        normalized = str(dataset).strip().casefold()
        candidate_id = self.datasets.get(normalized)
        if candidate_id is None:
            if self.fallback == "error":
                raise KeyError(
                    f"Dataset {dataset!r} is not configured in {self.source_file}; "
                    "fallback is explicitly set to 'error'."
                )
            candidate_id = self.fallback
        return self.resolve_candidate(normalized, candidate_id)


def _validate_lambda_value(candidate_id: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Candidate {candidate_id!r}: {key} must be numeric, got {value!r}."
        )
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(
            f"Candidate {candidate_id!r}: {key} must be finite and non-negative, "
            f"got {value!r}."
        )
    return converted


def load_lambda_configuration(path: str | Path) -> LambdaConfiguration:
    """Load the complete YAML and reject incomplete or ambiguous configurations."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"DMKCN lambda configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("DMKCN lambda configuration must be a YAML mapping.")
    observed_keys = set(payload)
    if observed_keys != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            "DMKCN lambda configuration must contain exactly "
            f"{sorted(EXPECTED_TOP_LEVEL_KEYS)}; observed {sorted(observed_keys)}."
        )
    if payload["schema_version"] != 1:
        raise ValueError(
            f"Unsupported lambda configuration schema: {payload['schema_version']!r}."
        )
    campaign_id = payload["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string.")

    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("candidates must be a non-empty mapping.")
    candidates: dict[str, tuple[float, float, float]] = {}
    seen_triplets: dict[tuple[float, float, float], str] = {}
    for candidate_id, values in raw_candidates.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Every candidate identifier must be a non-empty string.")
        if not isinstance(values, dict) or set(values) != set(LAMBDA_KEYS):
            raise ValueError(
                f"Candidate {candidate_id!r} must define exactly {list(LAMBDA_KEYS)}."
            )
        triplet = tuple(
            _validate_lambda_value(candidate_id, key, values[key])
            for key in LAMBDA_KEYS
        )
        if triplet in seen_triplets:
            raise ValueError(
                f"Candidates {seen_triplets[triplet]!r} and {candidate_id!r} "
                f"define the same lambda triplet {triplet}."
            )
        seen_triplets[triplet] = candidate_id
        candidates[candidate_id] = triplet

    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("datasets must be a non-empty mapping.")
    dataset_names = list(raw_datasets)
    expected_order = sorted(dataset_names, key=str.casefold)
    if dataset_names != expected_order:
        raise ValueError(
            f"datasets must be alphabetically ordered; observed {dataset_names}."
        )
    datasets: dict[str, str] = {}
    for dataset, candidate_id in raw_datasets.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("Every dataset identifier must be a non-empty string.")
        normalized = dataset.strip().casefold()
        if normalized in datasets:
            raise ValueError(f"Duplicate normalized dataset identifier: {dataset!r}.")
        if candidate_id not in candidates:
            raise ValueError(
                f"Dataset {dataset!r} references unknown candidate {candidate_id!r}."
            )
        datasets[normalized] = candidate_id

    fallback = payload["fallback"]
    if fallback != "error" and fallback not in candidates:
        raise ValueError(
            "fallback must be 'error' or an existing candidate identifier, "
            f"got {fallback!r}."
        )

    return LambdaConfiguration(
        campaign_id=campaign_id.strip(),
        candidates=candidates,
        datasets=datasets,
        fallback=fallback,
        source_file=str(config_path),
    )


def load_lambda_triplet(
    path: str | Path,
    dataset: str,
    candidate_id: str | None = None,
) -> LambdaTriplet:
    """Load and resolve one dataset triplet, optionally by candidate ID."""

    configuration = load_lambda_configuration(path)
    if candidate_id is None:
        return configuration.resolve(dataset)
    return configuration.resolve_candidate(dataset, candidate_id)
