"""Ternary protein signatures, strict batch alignment, and taxonomy building."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


UNKNOWN = "?"
POSITIVE = "+"
NEGATIVE = "-"
VALID_STATES = {UNKNOWN, POSITIVE, NEGATIVE}
DEFAULT_CONFIDENCE = 1.0


@dataclass(frozen=True)
class StateThresholds:
    """Thresholds used to turn posterior evidence into ``+``, ``-``, or ``?``."""

    min_marker_weight: float = 0.25
    min_state_confidence: float = 0.60
    min_state_separation: float = 0.15
    min_cluster_fraction: float = 0.60
    min_marker_effect: float = 0.10
    min_cell_state_probability: float = 0.60
    max_markers_per_cluster: int | None = 40
    max_markers_per_state: int | None = 20


def _marker_weights(marker_params: pd.DataFrame) -> np.ndarray:
    if "configuration_weight" in marker_params:
        values = marker_params["configuration_weight"]
    elif "phenotype_weight" in marker_params:
        values = marker_params["phenotype_weight"]
    elif "marker_weight" in marker_params:
        values = marker_params["marker_weight"]
    else:
        values = pd.Series(0.0, index=marker_params.index)
    return np.nan_to_num(
        pd.to_numeric(values, errors="coerce").to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clip(0.0, 1.0)


def _wilson_lower_bound(
    fraction: np.ndarray,
    n: int,
    *,
    z: float = 1.96,
) -> np.ndarray:
    """Conservative binomial support that accounts for cluster size."""

    values = np.asarray(fraction, dtype=float).clip(0.0, 1.0)
    if n <= 0:
        return np.zeros_like(values)
    denominator = 1.0 + z * z / n
    center = values + z * z / (2.0 * n)
    radius = z * np.sqrt(
        values * (1.0 - values) / n + z * z / (4.0 * n * n)
    )
    return np.clip((center - radius) / denominator, 0.0, 1.0)


def build_cluster_marker_states(
    *,
    batch_id: str,
    cluster_labels: np.ndarray,
    marker_names: Iterable[str],
    low_prob: np.ndarray,
    high_prob: np.ndarray,
    marker_params: pd.DataFrame,
    thresholds: StateThresholds = StateThresholds(),
) -> pd.DataFrame:
    """Build a complete long-form cluster-marker ternary configuration.

    Every state is called against the marker model fitted inside this batch.
    A marker that is not reliably bimodal in this batch remains unknown and
    contributes neither agreement nor conflict during alignment.
    """

    labels = np.asarray(cluster_labels, dtype=str)
    markers = np.asarray(list(marker_names), dtype=str)
    low = np.asarray(low_prob, dtype=float)
    high = np.asarray(high_prob, dtype=float)
    if low.shape != high.shape or low.shape != (len(labels), len(markers)):
        raise ValueError("low/high posterior matrices must be cells x markers")
    if len(marker_params) != len(markers):
        raise ValueError("marker_params must have exactly one row per marker")

    params = marker_params.reset_index(drop=True).copy()
    weights = _marker_weights(params)
    n_states = pd.to_numeric(
        params.get("n_states", pd.Series(2, index=params.index)),
        errors="coerce",
    ).fillna(1).to_numpy(int)
    status = params.get(
        "status", pd.Series("unknown", index=params.index)
    ).fillna("unknown").astype(str).to_numpy()
    reliable_status = np.array(
        [
            (s in {"valid_bimodal", "weak_bimodal", "unknown"})
            or s.startswith("valid")
            for s in status
        ],
        dtype=bool,
    )

    cell_conf = np.maximum(low, high)
    cell_state = np.where(high >= low, 1, 0)
    cell_state[cell_conf < thresholds.min_cell_state_probability] = -1
    records: list[dict] = []
    unique_clusters = sorted(
        np.unique(labels),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    for cluster in unique_clusters:
        inside = labels == str(cluster)
        outside = ~inside
        n_cells = int(inside.sum())
        mean_low = low[inside].mean(axis=0)
        mean_high = high[inside].mean(axis=0)
        dominant_idx = (mean_high >= mean_low).astype(int)
        dominant_prob = np.maximum(mean_low, mean_high)
        separation = np.abs(mean_high - mean_low)
        support_fraction = np.array(
            [
                np.mean(cell_state[inside, j] == dominant_idx[j])
                for j in range(len(markers))
            ],
            dtype=float,
        )
        support_lower = _wilson_lower_bound(support_fraction, n_cells)
        outside_low = low[outside].mean(axis=0) if outside.any() else np.zeros(len(markers))
        outside_high = high[outside].mean(axis=0) if outside.any() else np.zeros(len(markers))
        inside_dom = np.where(dominant_idx == 1, mean_high, mean_low)
        outside_dom = np.where(dominant_idx == 1, outside_high, outside_low)
        effect = (inside_dom - outside_dom) * weights
        high_a = high[inside].sum(axis=0) + 1.0
        high_b = low[inside].sum(axis=0) + 1.0
        high_dominance = 1.0 - beta_dist.cdf(0.5, high_a, high_b)
        dominance_posterior = np.where(
            dominant_idx == 1,
            high_dominance,
            1.0 - high_dominance,
        )

        identity_callable = (
            (n_states == 2)
            & reliable_status
            & (weights >= thresholds.min_marker_weight)
            & (dominant_prob >= thresholds.min_state_confidence)
            & (separation >= thresholds.min_state_separation)
            & (support_fraction >= thresholds.min_cluster_fraction)
            & (dominance_posterior >= 0.99)
        )
        evidence_score = (
            np.maximum(effect, 0.0)
            * dominant_prob
            * separation
            * support_fraction
        )
        call_confidence = weights * np.power(
            np.maximum(
                dominant_prob
                * separation
                * support_lower
                * dominance_posterior,
                0.0,
            ),
            0.25,
        )
        rankable = identity_callable & (effect >= thresholds.min_marker_effect)
        selected = np.flatnonzero(rankable)
        if thresholds.max_markers_per_state is not None:
            state_limited = []
            per_state = max(int(thresholds.max_markers_per_state), 0)
            for state_idx in (0, 1):
                candidates = selected[dominant_idx[selected] == state_idx]
                ranked = candidates[
                    np.argsort(-evidence_score[candidates], kind="stable")
                ]
                state_limited.extend(ranked[:per_state].tolist())
            selected = np.asarray(sorted(set(state_limited)), dtype=int)
        if thresholds.max_markers_per_cluster is not None:
            total = max(int(thresholds.max_markers_per_cluster), 0)
            selected = selected[
                np.argsort(-evidence_score[selected], kind="stable")
            ][:total]
        retained = np.zeros(len(markers), dtype=bool)
        retained[selected] = True
        calls = np.full(len(markers), UNKNOWN, dtype=object)
        calls[identity_callable & (dominant_idx == 1)] = POSITIVE
        calls[identity_callable & (dominant_idx == 0)] = NEGATIVE
        call_confidence[~identity_callable] = 0.0

        for j, marker in enumerate(markers):
            records.append(
                {
                    "batch": str(batch_id),
                    "cluster": str(cluster),
                    "local_node_id": f"{batch_id}|{cluster}",
                    "n_cells": n_cells,
                    "marker": str(marker),
                    "state": str(calls[j]),
                    "mean_low_prob": float(mean_low[j]),
                    "mean_high_prob": float(mean_high[j]),
                    "state_confidence": float(dominant_prob[j]),
                    "state_separation": float(separation[j]),
                    "cluster_state_fraction": float(support_fraction[j]),
                    "cluster_state_fraction_lower": float(support_lower[j]),
                    "marker_weight": float(weights[j]),
                    "marker_effect": float(effect[j]),
                    "dominance_posterior": float(dominance_posterior[j]),
                    "marker_n_states": int(n_states[j]),
                    "marker_status": str(status[j]),
                    "is_callable": bool(identity_callable[j]),
                    "is_retained": bool(retained[j]),
                    "evidence_score": float(evidence_score[j]),
                    "call_confidence": float(call_confidence[j]),
                }
            )
    return pd.DataFrame.from_records(records)


def signature_dicts(state_table: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Convert a long state table into node -> marker -> state mappings."""

    required = {"local_node_id", "marker", "state"}
    missing = required - set(state_table.columns)
    if missing:
        raise KeyError(f"state table missing columns: {sorted(missing)}")
    bad = set(state_table["state"].astype(str)) - VALID_STATES
    if bad:
        raise ValueError(f"invalid ternary states: {sorted(bad)}")
    return {
        str(node): dict(zip(sub["marker"].astype(str), sub["state"].astype(str)))
        for node, sub in state_table.groupby("local_node_id", sort=True)
    }


def confidence_dicts(
    state_table: pd.DataFrame,
    *,
    node_column: str = "local_node_id",
) -> dict[str, dict[str, float]]:
    """Return node-marker confidences, supporting legacy state-only tables."""

    if node_column not in state_table:
        raise KeyError(f"state table missing node column: {node_column}")
    confidence_column = next(
        (
            column
            for column in ("confidence", "call_confidence")
            if column in state_table
        ),
        None,
    )
    output: dict[str, dict[str, float]] = {}
    for node, sub in state_table.groupby(node_column, sort=True):
        states = sub["state"].astype(str).to_numpy()
        if confidence_column is None:
            values = np.where(states == UNKNOWN, 0.0, DEFAULT_CONFIDENCE)
        else:
            values = (
                pd.to_numeric(sub[confidence_column], errors="coerce")
                .fillna(0.0)
                .to_numpy(float)
                .clip(0.0, 1.0)
            )
            values[states == UNKNOWN] = 0.0
        output[str(node)] = dict(zip(sub["marker"].astype(str), values))
    return output


def alignment_marker_reliability(
    state_table: pd.DataFrame,
    *,
    minimum: float = 0.05,
) -> pd.DataFrame:
    """Report local configuration availability without cross-batch rescaling.

    Each batch has its own Gaussian model, so cross-batch posterior magnitudes
    are not used to alter marker weights. Unavailable states remain unknown and
    are ignored pairwise.
    """

    del minimum
    rows = []
    for marker, sub in state_table.groupby("marker", sort=True):
        callable_mask = sub["state"].astype(str).isin({POSITIVE, NEGATIVE})
        total_batches = sub["batch"].astype(str).nunique()
        callable_batches = sub.loc[callable_mask, "batch"].astype(str).nunique()
        rows.append(
            {
                "marker": str(marker),
                "n_batches": int(total_batches),
                "n_callable_batches": int(callable_batches),
                "n_local_clusters": int(len(sub)),
                "n_callable_local_clusters": int(callable_mask.sum()),
                "callable_batch_fraction": float(
                    callable_batches / max(total_batches, 1)
                ),
                "batch_eta_squared": np.nan,
                "alignment_reliability": 1.0,
                "batch_dominated": False,
            }
        )
    return pd.DataFrame(rows)


def _state_specificity(
    signatures: dict[str, dict[str, str]],
    *,
    minimum: float = 0.10,
) -> dict[tuple[str, str], float]:
    """Compute a bounded, data-driven rarity weight for each marker state."""

    floor = float(np.clip(minimum, 0.0, 1.0))
    n_nodes = len(signatures)
    if n_nodes == 0:
        return {}
    markers = sorted(set().union(*(signature for signature in signatures.values())))
    denominator = np.log(n_nodes + 1.0)
    output: dict[tuple[str, str], float] = {}
    for marker in markers:
        for state in (POSITIVE, NEGATIVE):
            count = sum(
                signature.get(marker, UNKNOWN) == state
                for signature in signatures.values()
            )
            rarity = (
                np.log((n_nodes + 1.0) / (count + 1.0)) / denominator
                if denominator > 0
                else 0.0
            )
            output[(marker, state)] = floor + (1.0 - floor) * float(rarity)
    return output


def _aggregate_compatible_signatures(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Union compatible evidence; raise if any descendant states conflict."""

    members = list(members)
    markers = sorted(set().union(*(signatures[node] for node in members)))
    output: dict[str, str] = {}
    for marker in markers:
        observed = {
            signatures[node].get(marker, UNKNOWN)
            for node in members
        } - {UNKNOWN}
        if observed == {POSITIVE, NEGATIVE}:
            raise ValueError(f"incompatible members at marker {marker}")
        output[marker] = next(iter(observed)) if observed else UNKNOWN
    return output


def _aggregate_compatible_confidences(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate aligned evidence and penalize missing member support."""

    members = list(members)
    markers = sorted(set().union(*(signatures[node] for node in members)))
    output: dict[str, float] = {}
    for marker in markers:
        observed = [
            confidences[node].get(marker, 0.0)
            for node in members
            if signatures[node].get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        ]
        coverage = len(observed) / len(members)
        output[marker] = (
            float(np.median(observed) * coverage) if observed else 0.0
        )
    return output


def _pair_evidence(
    left: dict[str, str],
    right: dict[str, str],
    left_confidence: dict[str, float] | None = None,
    right_confidence: dict[str, float] | None = None,
    specificity: dict[tuple[str, str], float] | None = None,
) -> dict:
    left_confidence = left_confidence or {}
    right_confidence = right_confidence or {}
    specificity = specificity or {}
    markers = sorted(set(left) | set(right))
    shared_pos = [
        marker for marker in markers
        if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN) == POSITIVE
    ]
    shared_neg = [
        marker for marker in markers
        if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN) == NEGATIVE
    ]
    conflicts = [
        marker for marker in markers
        if {left.get(marker, UNKNOWN), right.get(marker, UNKNOWN)}
        == {POSITIVE, NEGATIVE}
    ]
    shared_observed = shared_pos + shared_neg
    observed_left = sum(
        left.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        for marker in markers
    )
    observed_right = sum(
        right.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        for marker in markers
    )
    jointly_observed = len(shared_observed) + len(conflicts)

    def common_weight(marker: str, state: str) -> float:
        confidence = min(
            left_confidence.get(marker, DEFAULT_CONFIDENCE),
            right_confidence.get(marker, DEFAULT_CONFIDENCE),
        )
        return confidence * specificity.get((marker, state), DEFAULT_CONFIDENCE)

    weighted_pos = sum(common_weight(marker, POSITIVE) for marker in shared_pos)
    weighted_neg = sum(common_weight(marker, NEGATIVE) for marker in shared_neg)
    weighted_conflicts = 0.0
    for marker in conflicts:
        confidence = min(
            left_confidence.get(marker, DEFAULT_CONFIDENCE),
            right_confidence.get(marker, DEFAULT_CONFIDENCE),
        )
        state_weight = max(
            specificity.get((marker, POSITIVE), DEFAULT_CONFIDENCE),
            specificity.get((marker, NEGATIVE), DEFAULT_CONFIDENCE),
        )
        weighted_conflicts += confidence * state_weight
    return {
        "n_shared_observed": len(shared_observed),
        "n_observed_left": int(observed_left),
        "n_observed_right": int(observed_right),
        "n_jointly_observed": int(jointly_observed),
        "configuration_coverage_left": float(
            jointly_observed / max(observed_left, 1)
        ),
        "configuration_coverage_right": float(
            jointly_observed / max(observed_right, 1)
        ),
        "minimum_configuration_coverage": float(
            min(
                jointly_observed / max(observed_left, 1),
                jointly_observed / max(observed_right, 1),
            )
        ),
        "n_common_positive": len(shared_pos),
        "n_common_negative": len(shared_neg),
        "n_conflicts": len(conflicts),
        "weighted_shared_observed": weighted_pos + weighted_neg,
        "weighted_common_positive": weighted_pos,
        "weighted_common_negative": weighted_neg,
        "weighted_conflicts": weighted_conflicts,
        "shared_positive_markers": ";".join(shared_pos),
        "shared_negative_markers": ";".join(shared_neg),
        "conflict_markers": ";".join(conflicts),
    }


def _unique_best_partners(
    group_ids: list[str],
    candidate_scores: dict[tuple[str, str], tuple],
    *,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.02,
) -> dict[str, str]:
    """Return only candidates whose score is uniquely best for each group."""

    def compare(left: tuple, right: tuple) -> int:
        for left_value, right_value in zip(left, right):
            tolerance = max(
                float(absolute_tolerance),
                float(relative_tolerance)
                * max(abs(float(left_value)), abs(float(right_value))),
            )
            difference = float(left_value) - float(right_value)
            if difference > tolerance:
                return 1
            if difference < -tolerance:
                return -1
        return 0

    output: dict[str, str] = {}
    for group in group_ids:
        candidates = []
        for pair, score in candidate_scores.items():
            if group not in pair:
                continue
            other = pair[1] if pair[0] == group else pair[0]
            candidates.append((other, score))
        if not candidates:
            continue
        best = [
            (other, score)
            for other, score in candidates
            if all(
                other == competitor
                or compare(score, competitor_score) > 0
                for competitor, competitor_score in candidates
            )
        ]
        if len(best) == 1:
            output[group] = best[0][0]
    return output


def _robust_group_profile(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    batches: dict[str, str],
    *,
    consensus_fraction: float = 2.0 / 3.0,
) -> tuple[dict[str, str], dict[str, float]]:
    """Estimate a batch-robust prototype without propagating one-off states."""

    members = list(members)
    n_batches = len({batches[node] for node in members})
    required_batches = 1 if n_batches == 1 else 2
    markers = sorted(set().union(*(signatures[node] for node in members)))
    profile: dict[str, str] = {}
    profile_confidence: dict[str, float] = {}
    for marker in markers:
        votes = {POSITIVE: 0.0, NEGATIVE: 0.0}
        observed_batches: set[str] = set()
        state_confidences = {POSITIVE: [], NEGATIVE: []}
        for node in members:
            state = signatures[node].get(marker, UNKNOWN)
            if state not in {POSITIVE, NEGATIVE}:
                continue
            confidence = float(confidences[node].get(marker, 0.0))
            votes[state] += confidence
            state_confidences[state].append(confidence)
            observed_batches.add(batches[node])
        total = votes[POSITIVE] + votes[NEGATIVE]
        if len(observed_batches) < required_batches or total <= 0:
            profile[marker] = UNKNOWN
            profile_confidence[marker] = 0.0
            continue
        state = max((POSITIVE, NEGATIVE), key=lambda value: votes[value])
        agreement = votes[state] / total
        if agreement < consensus_fraction:
            profile[marker] = UNKNOWN
            profile_confidence[marker] = 0.0
            continue
        coverage = len(observed_batches) / max(n_batches, 1)
        profile[marker] = state
        profile_confidence[marker] = float(
            np.median(state_confidences[state])
            * agreement
            * (0.5 + 0.5 * coverage)
        )
    return profile, profile_confidence


def _alignment_candidate_pairs(
    nodes: list[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    specificity: dict[tuple[str, str], float],
    batches: dict[str, str],
    *,
    max_neighbors: int | None = None,
    chunk_size: int = 512,
) -> set[tuple[str, str]]:
    """Generate cross-batch candidates using chunked signed-state similarity."""

    if len(nodes) < 2:
        return set()
    markers = sorted(set().union(*(signatures[node] for node in nodes)))
    marker_index = {marker: index for index, marker in enumerate(markers)}
    matrix = np.zeros((len(nodes), len(markers)), dtype=np.float32)
    for row, node in enumerate(nodes):
        for marker, state in signatures[node].items():
            if state not in {POSITIVE, NEGATIVE}:
                continue
            confidence = max(float(confidences[node].get(marker, 0.0)), 0.0)
            state_weight = max(float(specificity.get((marker, state), 1.0)), 0.0)
            sign = 1.0 if state == POSITIVE else -1.0
            matrix[row, marker_index[marker]] = sign * np.sqrt(
                confidence * state_weight
            )
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] /= norms[nonzero, None]
    n_batches = len(set(batches.values()))
    if max_neighbors is None:
        max_neighbors = max(32, min(128, 2 * n_batches))
    k = min(max(int(max_neighbors), 1), len(nodes) - 1)
    batch_array = np.asarray([batches[node] for node in nodes], dtype=object)
    pairs: set[tuple[str, str]] = set()
    for start in range(0, len(nodes), max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), len(nodes))
        similarity = matrix[start:stop] @ matrix.T
        similarity[batch_array[start:stop, None] == batch_array[None, :]] = -np.inf
        local_rows = np.arange(stop - start)
        similarity[local_rows, np.arange(start, stop)] = -np.inf
        if k < len(nodes):
            neighbor_idx = np.argpartition(similarity, -k, axis=1)[:, -k:]
        else:
            neighbor_idx = np.tile(np.arange(len(nodes)), (stop - start, 1))
        for local_row, candidates in enumerate(neighbor_idx):
            source = start + local_row
            for target in candidates:
                if not np.isfinite(similarity[local_row, target]):
                    continue
                left, right = sorted((nodes[source], nodes[int(target)]))
                pairs.add((left, right))
    return pairs


def _alignment_evidence(
    left: str,
    right: str,
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    specificity: dict[tuple[str, str], float],
    *,
    use_weighted_scores: bool,
) -> dict:
    evidence = _pair_evidence(
        signatures[left],
        signatures[right],
        confidences[left] if use_weighted_scores else None,
        confidences[right] if use_weighted_scores else None,
        specificity if use_weighted_scores else None,
    )
    conflict_markers = [
        marker for marker in evidence["conflict_markers"].split(";") if marker
    ]
    strong_conflicts = sum(
        min(
            confidences[left].get(marker, DEFAULT_CONFIDENCE),
            confidences[right].get(marker, DEFAULT_CONFIDENCE),
        )
        >= 0.80
        for marker in conflict_markers
    )
    shared = (
        evidence["weighted_shared_observed"]
        if use_weighted_scores
        else float(evidence["n_shared_observed"])
    )
    conflicts = (
        evidence["weighted_conflicts"]
        if use_weighted_scores
        else float(evidence["n_conflicts"])
    )
    agreement = shared / max(shared + conflicts, 1e-12)
    evidence_strength = 1.0 - np.exp(-shared / 3.0)
    evidence.update(
        {
            "n_strong_conflicts": int(strong_conflicts),
            "agreement_fraction": float(agreement),
            "alignment_quality": float(agreement * evidence_strength),
        }
    )
    return evidence


def _batch_conflicts_are_tolerable(
    evidence: dict,
    *,
    max_batch_conflicts: int,
    allow_strong_conflicts: bool = False,
) -> bool:
    """Allow a small number of weak cross-batch state disagreements."""

    return (
        int(evidence["n_conflicts"]) <= max(int(max_batch_conflicts), 0)
        and (
            bool(allow_strong_conflicts)
            or int(evidence["n_strong_conflicts"]) == 0
        )
    )


def align_clusters(
    state_table: pd.DataFrame,
    *,
    min_shared_observed: int = 3,
    minimum_state_specificity: float = 0.10,
    score_absolute_tolerance: float = 0.05,
    score_relative_tolerance: float = 0.02,
    use_weighted_scores: bool = True,
    min_alignment_agreement: float = 0.80,
    min_configuration_coverage: float = 0.60,
    min_prototype_batches: int = 2,
    max_candidate_neighbors: int | None = None,
    max_batch_conflicts: int = 2,
    allow_strong_conflicts: bool = False,
) -> dict[str, pd.DataFrame]:
    """Align independently fitted configurations across batches.

    Unknown states encode unavailable resolving ability. They are never
    treated as matches, and asymmetric coarse/fine configurations remain
    unresolved unless enough of both configurations is jointly observable.
    Up to ``max_batch_conflicts`` weak disagreements may be tolerated for
    otherwise near-best cross-batch matches; strong state conflicts remain a
    hard boundary.
    """

    if not 0.0 <= float(min_configuration_coverage) <= 1.0:
        raise ValueError("min_configuration_coverage must be between 0 and 1")

    meta = (
        state_table[["local_node_id", "batch", "cluster", "n_cells"]]
        .drop_duplicates("local_node_id")
        .set_index("local_node_id")
    )
    signatures = signature_dicts(state_table)
    confidences = confidence_dicts(state_table)
    marker_reliability = alignment_marker_reliability(state_table)
    reliability_map = marker_reliability.set_index("marker")[
        "alignment_reliability"
    ].to_dict()
    for node in confidences:
        for marker in confidences[node]:
            confidences[node][marker] *= float(reliability_map.get(marker, 1.0))
    specificity = (
        _state_specificity(signatures, minimum=minimum_state_specificity)
        if use_weighted_scores
        else {}
    )
    nodes = sorted(signatures)
    batch_by_node = meta["batch"].astype(str).to_dict()
    candidate_pairs = _alignment_candidate_pairs(
        nodes,
        signatures,
        confidences,
        specificity,
        batch_by_node,
        max_neighbors=max_candidate_neighbors,
    )
    pair_evidence = {
        pair: _alignment_evidence(
            pair[0],
            pair[1],
            signatures,
            confidences,
            specificity,
            use_weighted_scores=use_weighted_scores,
        )
        for pair in sorted(candidate_pairs)
    }
    eligible_pairs = {
        pair: evidence
        for pair, evidence in pair_evidence.items()
        if evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
        and _batch_conflicts_are_tolerable(
            evidence,
            max_batch_conflicts=max_batch_conflicts,
            allow_strong_conflicts=allow_strong_conflicts,
        )
        and evidence["agreement_fraction"] >= min_alignment_agreement
        and evidence["minimum_configuration_coverage"]
        >= min_configuration_coverage
    }
    best_quality = {node: -np.inf for node in nodes}
    for (left, right), evidence in eligible_pairs.items():
        quality = evidence["alignment_quality"]
        best_quality[left] = max(best_quality[left], quality)
        best_quality[right] = max(best_quality[right], quality)

    parent = {node: node for node in nodes}
    members = {node: {node} for node in nodes}
    component_batches = {node: {batch_by_node[node]} for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if not component_batches[left_root].isdisjoint(component_batches[right_root]):
            return False
        proposed = members[left_root] | members[right_root]
        prototype_conflicts = 0
        for marker in sorted(
            set().union(*(signatures[node] for node in proposed))
        ):
            if reliability_map.get(marker, 1.0) < 0.25:
                continue
            votes = {POSITIVE: 0.0, NEGATIVE: 0.0}
            for node in proposed:
                state = signatures[node].get(marker, UNKNOWN)
                if state in votes:
                    votes[state] += confidences[node].get(marker, 0.0)
            total = votes[POSITIVE] + votes[NEGATIVE]
            if (
                min(votes.values()) >= 0.80
                and max(votes.values()) / total < 2.0 / 3.0
            ):
                prototype_conflicts += 1
        if prototype_conflicts and not allow_strong_conflicts:
            return False
        if prototype_conflicts > max(int(max_batch_conflicts), 0):
            return False
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        component_batches[left_root].update(component_batches.pop(right_root))
        return True

    tolerance = max(float(score_absolute_tolerance), 0.0)
    audit_rows: list[dict] = []
    seed_pairs = []
    for pair, evidence in eligible_pairs.items():
        left, right = pair
        quality = evidence["alignment_quality"]
        near_best_left = quality >= best_quality[left] - tolerance
        near_best_right = quality >= best_quality[right] - tolerance
        if near_best_left and near_best_right:
            seed_pairs.append(pair)
    seed_pairs.sort(
        key=lambda pair: pair_evidence[pair]["alignment_quality"],
        reverse=True,
    )
    selected_seed: set[tuple[str, str]] = set()
    for left, right in seed_pairs:
        if union(left, right):
            selected_seed.add((left, right))

    for pair, evidence in pair_evidence.items():
        audit_rows.append(
            {
                "round": 1,
                "stage": "seed",
                "group_a": pair[0],
                "group_b": pair[1],
                "disjoint_batches": batch_by_node[pair[0]] != batch_by_node[pair[1]],
                "eligible": pair in eligible_pairs,
                "score": repr(
                    (
                        evidence["alignment_quality"],
                        evidence["weighted_shared_observed"],
                    )
                ),
                "selected_for_merge": pair in selected_seed,
                "supporting_batches": 0,
                **evidence,
            }
        )

    rescue_round = 1
    while True:
        rescue_round += 1
        roots = sorted({find(node) for node in nodes})
        groups = {root: tuple(sorted(members[root])) for root in roots}
        profiles = {}
        profile_confidences = {}
        for root, group_members in groups.items():
            profile, confidence = _robust_group_profile(
                group_members,
                signatures,
                confidences,
                batch_by_node,
            )
            profiles[root] = profile
            profile_confidences[root] = confidence

        component_candidates: set[tuple[str, str]] = set()
        for left, right in candidate_pairs:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                component_candidates.add(tuple(sorted((left_root, right_root))))
        rescue_evidence: dict[tuple[str, str], dict] = {}
        for left_root, right_root in sorted(component_candidates):
            if not component_batches[left_root].isdisjoint(
                component_batches[right_root]
            ):
                continue
            evidence = _pair_evidence(
                profiles[left_root],
                profiles[right_root],
                profile_confidences[left_root],
                profile_confidences[right_root],
                specificity,
            )
            shared = evidence["weighted_shared_observed"]
            conflicts = evidence["weighted_conflicts"]
            agreement = shared / max(shared + conflicts, 1e-12)
            strong_conflicts = sum(
                min(
                    profile_confidences[left_root].get(marker, 0.0),
                    profile_confidences[right_root].get(marker, 0.0),
                )
                >= 0.80
                for marker in evidence["conflict_markers"].split(";")
                if marker
            )
            supporting_left: set[str] = set()
            supporting_right: set[str] = set()
            for left_node in groups[left_root]:
                for right_node in groups[right_root]:
                    pair = tuple(sorted((left_node, right_node)))
                    local = pair_evidence.get(pair)
                    if (
                        local is not None
                        and local["n_shared_observed"] >= max(int(min_shared_observed), 1)
                        and local["agreement_fraction"] >= min_alignment_agreement
                        and _batch_conflicts_are_tolerable(
                            local,
                            max_batch_conflicts=max_batch_conflicts,
                            allow_strong_conflicts=allow_strong_conflicts,
                        )
                        and local["minimum_configuration_coverage"]
                        >= min_configuration_coverage
                    ):
                        supporting_left.add(batch_by_node[left_node])
                        supporting_right.add(batch_by_node[right_node])
            required_left = min(
                max(int(min_prototype_batches), 1),
                len(component_batches[left_root]),
            )
            required_right = min(
                max(int(min_prototype_batches), 1),
                len(component_batches[right_root]),
            )
            eligible = (
                evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
                and agreement >= min_alignment_agreement
                and _batch_conflicts_are_tolerable(
                    {**evidence, "n_strong_conflicts": strong_conflicts},
                    # Group prototypes can accumulate disagreements through
                    # transitivity, so their rescue budget is deliberately
                    # tighter than the direct cross-batch matching budget.
                    max_batch_conflicts=min(max_batch_conflicts, 1),
                    allow_strong_conflicts=allow_strong_conflicts,
                )
                and evidence["minimum_configuration_coverage"]
                >= min_configuration_coverage
                and len(supporting_left) >= required_left
                and len(supporting_right) >= required_right
            )
            quality = agreement * (
                1.0 - np.exp(-shared / 3.0)
            )
            rescue_evidence[(left_root, right_root)] = {
                **evidence,
                "n_strong_conflicts": int(strong_conflicts),
                "agreement_fraction": float(agreement),
                "alignment_quality": float(quality),
                "supporting_batches": len(supporting_left | supporting_right),
                "supporting_batches_a": len(supporting_left),
                "supporting_batches_b": len(supporting_right),
                "eligible": bool(eligible),
            }

        best_for_root = {root: -np.inf for root in roots}
        for (left_root, right_root), evidence in rescue_evidence.items():
            if not evidence["eligible"]:
                continue
            quality = evidence["alignment_quality"]
            best_for_root[left_root] = max(best_for_root[left_root], quality)
            best_for_root[right_root] = max(best_for_root[right_root], quality)
        merge_candidates = []
        for pair, evidence in rescue_evidence.items():
            if not evidence["eligible"]:
                continue
            left_root, right_root = pair
            quality = evidence["alignment_quality"]
            left_multi = len(component_batches[left_root]) >= min_prototype_batches
            right_multi = len(component_batches[right_root]) >= min_prototype_batches
            left_best = quality >= best_for_root[left_root] - tolerance
            right_best = quality >= best_for_root[right_root] - tolerance
            if (left_best and right_best) or (
                left_best and right_multi and not left_multi
            ) or (right_best and left_multi and not right_multi):
                merge_candidates.append(pair)
        merge_candidates.sort(
            key=lambda pair: rescue_evidence[pair]["alignment_quality"],
            reverse=True,
        )
        selected_rescue: set[tuple[str, str]] = set()
        for left_root, right_root in merge_candidates:
            if union(left_root, right_root):
                selected_rescue.add((left_root, right_root))
        for pair, evidence in rescue_evidence.items():
            audit_rows.append(
                {
                    "round": rescue_round,
                    "stage": "prototype",
                    "group_a": pair[0],
                    "group_b": pair[1],
                    "disjoint_batches": True,
                    "score": repr(
                        (
                            evidence["alignment_quality"],
                            evidence["weighted_shared_observed"],
                        )
                    ),
                    "selected_for_merge": pair in selected_rescue,
                    **evidence,
                }
            )
        if not selected_rescue:
            break

    final_groups = {
        root: tuple(sorted(members[root]))
        for root in sorted({find(node) for node in nodes})
    }
    ordered_groups = sorted(
        final_groups.values(),
        key=lambda members: (-len(members), members),
    )
    membership_rows = []
    state_rows = []
    for index, members in enumerate(ordered_groups, start=1):
        aligned_id = f"ALIGNED_{index:04d}"
        aggregate, aggregate_confidence = _robust_group_profile(
            members,
            signatures,
            confidences,
            batch_by_node,
        )
        for node in members:
            row = meta.loc[node]
            membership_rows.append(
                {
                    "aligned_node_id": aligned_id,
                    "local_node_id": node,
                    "batch": str(row["batch"]),
                    "cluster": str(row["cluster"]),
                    "n_cells": int(row["n_cells"]),
                }
            )
        for marker, state in sorted(aggregate.items()):
            observed_members = sum(
                signatures[node].get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
                for node in members
            )
            state_rows.append(
                {
                    "aligned_node_id": aligned_id,
                    "marker": marker,
                    "state": state,
                    "confidence": aggregate_confidence[marker],
                    "observed_member_fraction": observed_members / len(members),
                    "state_specificity": (
                        specificity.get((marker, state), 0.0)
                        if state in {POSITIVE, NEGATIVE}
                        else 0.0
                    ),
                }
            )
    return {
        "membership": pd.DataFrame(membership_rows),
        "states": pd.DataFrame(state_rows),
        "pair_audit": pd.DataFrame(audit_rows),
        "marker_reliability": marker_reliability,
    }


def _intersect_parent_state(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, str]:
    markers = sorted(set(left) | set(right))
    return {
        marker: (
            left.get(marker, UNKNOWN)
            if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN)
            and left.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
            else UNKNOWN
        )
        for marker in markers
    }


def build_taxonomy(
    aligned_membership: pd.DataFrame,
    aligned_states: pd.DataFrame,
    *,
    min_tree_shared_observed: int = 1,
    min_tree_weighted_shared: float = 0.0,
    max_tree_conflicts: int | None = None,
    minimum_state_specificity: float = 0.10,
    score_absolute_tolerance: float = 0.05,
    score_relative_tolerance: float = 0.02,
    use_weighted_scores: bool = True,
) -> dict[str, pd.DataFrame | dict]:
    """Build a bottom-up forest with lexicographic, ambiguity-safe merging."""

    leaf_signatures = {
        str(node): dict(zip(sub["marker"].astype(str), sub["state"].astype(str)))
        for node, sub in aligned_states.groupby("aligned_node_id", sort=True)
    }
    leaf_confidences = confidence_dicts(
        aligned_states,
        node_column="aligned_node_id",
    )
    specificity = (
        _state_specificity(leaf_signatures, minimum=minimum_state_specificity)
        if use_weighted_scores
        else {}
    )
    membership = {
        str(node): tuple(sorted(sub["local_node_id"].astype(str)))
        for node, sub in aligned_membership.groupby("aligned_node_id", sort=True)
    }
    active = sorted(leaf_signatures)
    signatures = dict(leaf_signatures)
    confidences = dict(leaf_confidences)
    descendants = {node: (node,) for node in active}
    node_kind = {node: "aligned_leaf" for node in active}
    edges: list[dict] = []
    merges: list[dict] = []
    pair_audit: list[dict] = []
    next_internal = 1
    round_id = 0

    while len(active) > 1:
        round_id += 1
        scores: dict[tuple[str, str], tuple] = {}
        evidence_by_pair: dict[tuple[str, str], dict] = {}
        for left, right in itertools.combinations(sorted(active), 2):
            evidence = _pair_evidence(
                signatures[left],
                signatures[right],
                confidences[left] if use_weighted_scores else None,
                confidences[right] if use_weighted_scores else None,
                specificity if use_weighted_scores else None,
            )
            evidence_by_pair[(left, right)] = evidence
            if evidence["n_shared_observed"] < max(
                int(min_tree_shared_observed),
                1,
            ):
                continue
            if (
                use_weighted_scores
                and evidence["weighted_shared_observed"]
                < max(float(min_tree_weighted_shared), 0.0)
            ):
                continue
            if (
                max_tree_conflicts is not None
                and evidence["n_conflicts"] > max(int(max_tree_conflicts), 0)
            ):
                continue
            score = (
                evidence["weighted_common_positive"],
                evidence["weighted_common_negative"],
                -evidence["weighted_conflicts"],
            )
            scores[(left, right)] = score

        unique_best = _unique_best_partners(
            sorted(active),
            scores,
            absolute_tolerance=score_absolute_tolerance if use_weighted_scores else 0.0,
            relative_tolerance=score_relative_tolerance if use_weighted_scores else 0.0,
        )
        merge_pairs = sorted(
            {
                tuple(sorted((left, right)))
                for left, right in unique_best.items()
                if unique_best.get(right) == left
            }
        )
        for pair, evidence in evidence_by_pair.items():
            score = scores.get(pair)
            pair_audit.append(
                {
                    "round": round_id,
                    "node_a": pair[0],
                    "node_b": pair[1],
                    "score": repr(score) if score is not None else "",
                    "eligible": score is not None,
                    "is_unique_best_for_a": unique_best.get(pair[0]) == pair[1],
                    "is_unique_best_for_b": unique_best.get(pair[1]) == pair[0],
                    "selected_for_merge": pair in merge_pairs,
                    **evidence,
                }
            )
        if not merge_pairs:
            break

        merged_nodes = set(itertools.chain.from_iterable(merge_pairs))
        new_active = [node for node in active if node not in merged_nodes]
        for left, right in merge_pairs:
            parent = f"TAXON_{next_internal:04d}"
            next_internal += 1
            evidence = evidence_by_pair[(left, right)]
            signatures[parent] = _intersect_parent_state(
                signatures[left], signatures[right]
            )
            confidences[parent] = {
                marker: (
                    min(
                        confidences[left].get(marker, 0.0),
                        confidences[right].get(marker, 0.0),
                    )
                    if state in {POSITIVE, NEGATIVE}
                    else 0.0
                )
                for marker, state in signatures[parent].items()
            }
            descendants[parent] = tuple(
                sorted(descendants[left] + descendants[right])
            )
            node_kind[parent] = "internal"
            edges.extend(
                [
                    {"parent": parent, "child": left},
                    {"parent": parent, "child": right},
                ]
            )
            merges.append(
                {
                    "round": round_id,
                    "parent": parent,
                    "child_a": left,
                    "child_b": right,
                    "score": repr(
                        (
                            evidence["weighted_common_positive"],
                            evidence["weighted_common_negative"],
                            -evidence["weighted_conflicts"],
                        )
                    ),
                    **evidence,
                }
            )
            new_active.append(parent)
        active = sorted(new_active)

    roots = sorted(active)
    all_local_members = {
        node: tuple(
            sorted(
                itertools.chain.from_iterable(
                    membership[leaf] for leaf in descendants[node]
                )
            )
        )
        for node in signatures
    }
    node_rows = []
    state_rows = []
    for node in sorted(signatures):
        sig = signatures[node]
        node_rows.append(
            {
                "tree_node_id": node,
                "node_kind": node_kind[node],
                "is_root": node in roots,
                "n_aligned_leaves": len(descendants[node]),
                "n_local_clusters": len(all_local_members[node]),
                "n_positive_markers": sum(x == POSITIVE for x in sig.values()),
                "n_negative_markers": sum(x == NEGATIVE for x in sig.values()),
                "n_unknown_markers": sum(x == UNKNOWN for x in sig.values()),
                "mean_observed_confidence": float(
                    np.mean(
                        [
                            confidences[node].get(marker, 0.0)
                            for marker, state in sig.items()
                            if state in {POSITIVE, NEGATIVE}
                        ]
                    )
                )
                if any(state in {POSITIVE, NEGATIVE} for state in sig.values())
                else 0.0,
                "aligned_leaf_descendants": ";".join(descendants[node]),
                "local_cluster_descendants": ";".join(all_local_members[node]),
            }
        )
        for marker, state in sorted(sig.items()):
            state_rows.append(
                {
                    "tree_node_id": node,
                    "marker": marker,
                    "state": state,
                    "confidence": confidences[node].get(marker, 0.0),
                    "state_specificity": (
                        specificity.get((marker, state), 0.0)
                        if state in {POSITIVE, NEGATIVE}
                        else 0.0
                    ),
                }
            )

    forest = {
        "roots": roots,
        "children": {
            node: [
                edge["child"] for edge in edges if edge["parent"] == node
            ]
            for node in sorted(signatures)
            if any(edge["parent"] == node for edge in edges)
        },
    }
    return {
        "nodes": pd.DataFrame(node_rows),
        "edges": pd.DataFrame(edges, columns=["parent", "child"]),
        "merges": pd.DataFrame(merges),
        "states": pd.DataFrame(state_rows),
        "pair_audit": pd.DataFrame(pair_audit),
        "forest": forest,
    }


def build_cell_tree_assignments(
    cell_clusters: pd.DataFrame,
    aligned_membership: pd.DataFrame,
    tree_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Attach aligned leaves and root-to-leaf taxonomy paths to cells."""

    cells = cell_clusters.copy()
    cells["local_node_id"] = (
        cells["batch"].astype(str) + "|" + cells["protein_phenotype"].astype(str)
    )
    leaf_map = aligned_membership.set_index("local_node_id")[
        "aligned_node_id"
    ].astype(str)
    cells["aligned_leaf_id"] = cells["local_node_id"].map(leaf_map)
    parent_of = (
        tree_edges.set_index("child")["parent"].astype(str).to_dict()
        if not tree_edges.empty
        else {}
    )

    def path_for(leaf: str) -> str:
        if pd.isna(leaf):
            return ""
        path = [str(leaf)]
        seen = set(path)
        while path[-1] in parent_of:
            parent = parent_of[path[-1]]
            if parent in seen:
                raise ValueError("cycle detected in taxonomy")
            path.append(parent)
            seen.add(parent)
        return ";".join(reversed(path))

    cells["taxonomy_path"] = cells["aligned_leaf_id"].map(path_for)
    cells["taxonomy_root"] = cells["taxonomy_path"].str.split(";").str[0]
    return cells
