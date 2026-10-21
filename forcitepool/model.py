"""Cross-batch shared marker-state reference with batch-local clustering."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cytofuse.methods.paramshared import (
    EPS,
    BaseCluster,
    MarkerModelSelector,
    ParamSharedCluster,
    add_gaussian_jitter,
    robust_zscore_per_marker,
)


DEFAULT_TECHNICAL_PATTERNS = (
    "isotype",
    "isotyp",
    "ctrl",
    "control",
)
DEFAULT_TECHNICAL_PREFIXES = ("mouse",)


def technical_marker_mask(
    marker_names: Iterable[str],
    *,
    patterns: tuple[str, ...] = DEFAULT_TECHNICAL_PATTERNS,
    prefixes: tuple[str, ...] = DEFAULT_TECHNICAL_PREFIXES,
) -> np.ndarray:
    """Identify explicit assay-control markers using conservative name rules."""

    output = []
    for marker in marker_names:
        normalized = str(marker).strip().lower().replace("_", "")
        output.append(
            any(pattern in normalized for pattern in patterns)
            or any(normalized.startswith(prefix) for prefix in prefixes)
        )
    return np.asarray(output, dtype=bool)


@dataclass
class SharedMarkerReference:
    """Frozen marker distributions estimated from a balanced cross-batch pool."""

    marker_names: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    base_cluster: BaseCluster
    normalization_mode: str = "global"

    def validate(self) -> None:
        n_markers = len(self.marker_names)
        if n_markers == 0:
            raise ValueError("shared marker reference has no markers")
        if self.center.shape != (n_markers,) or self.scale.shape != (n_markers,):
            raise ValueError("reference center/scale do not match marker names")
        if self.base_cluster.n_markers != n_markers:
            raise ValueError("reference BaseCluster does not match marker names")
        if np.any(~np.isfinite(self.center)):
            raise ValueError("reference center contains non-finite values")
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("reference scale must be finite and positive")
        if self.normalization_mode not in {"global", "batch_robust"}:
            raise ValueError("unknown shared-reference normalization mode")
        if not np.array_equal(
            self.marker_names.astype(str),
            self.base_cluster.marker_names.astype(str),
        ):
            raise ValueError("reference marker order is inconsistent")

    def transform_with_diagnostics(
        self,
        matrix: np.ndarray,
        marker_names: Iterable[str],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        names = np.asarray(list(marker_names), dtype=str)
        if not np.array_equal(names, self.marker_names.astype(str)):
            raise ValueError("batch marker names/order differ from shared reference")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("matrix must be cells x reference markers")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("batch matrix contains non-finite values")
        if self.normalization_mode != "batch_robust":
            transformed = (matrix - self.center) / self.scale
            diagnostics = pd.DataFrame(
                {
                    "marker": names,
                    "batch_calibration_status": "global_reference",
                    "batch_calibration_scale": 1.0,
                    "batch_calibration_shift": 0.0,
                }
            )
            return transformed, diagnostics

        transformed, _, _ = robust_zscore_per_marker(matrix)
        calibrated = transformed.copy()
        selector = MarkerModelSelector(
            random_state=0,
            n_init=3,
            min_component_weight=0.025,
            bic_gain_12=10.0,
            min_separation=1.0,
            min_mean_posterior_conf=0.70,
            max_density_valley_ratio=0.90,
        )
        rows = []
        for marker_idx, marker in enumerate(names):
            status = "not_shared_bimodal"
            scale_factor = 1.0
            shift = 0.0
            local_low = np.nan
            local_high = np.nan
            if self.base_cluster.n_states[marker_idx] == 2:
                local_fit, metrics = selector.select_marker_state_count(
                    transformed[:, marker_idx]
                )
                status = str(metrics["status"])
                if status == "valid_bimodal" and local_fit.n_states == 2:
                    shared_mu = np.asarray(
                        self.base_cluster.mu[marker_idx],
                        dtype=float,
                    )
                    local_low, local_high = map(float, local_fit.mu[[0, -1]])
                    local_span = local_high - local_low
                    shared_span = float(shared_mu[-1] - shared_mu[0])
                    if local_span > 1e-8 and shared_span > 1e-8:
                        raw_scale = shared_span / local_span
                        if 0.33 <= raw_scale <= 3.0:
                            local_mid = 0.5 * (local_low + local_high)
                            shared_mid = 0.5 * (shared_mu[0] + shared_mu[-1])
                            scale_factor = float(raw_scale)
                            shift = float(shared_mid - scale_factor * local_mid)
                            calibrated[:, marker_idx] = (
                                scale_factor * transformed[:, marker_idx] + shift
                            )
                            status = "calibrated_two_peaks"
                        else:
                            status = "unstable_local_scale"
            rows.append(
                {
                    "marker": str(marker),
                    "batch_calibration_status": status,
                    "batch_calibration_scale": scale_factor,
                    "batch_calibration_shift": shift,
                    "batch_local_low_center": local_low,
                    "batch_local_high_center": local_high,
                }
            )
        return calibrated, pd.DataFrame(rows)

    def transform(self, matrix: np.ndarray, marker_names: Iterable[str]) -> np.ndarray:
        transformed, _ = self.transform_with_diagnostics(matrix, marker_names)
        return transformed

    def fresh_base_cluster(self) -> BaseCluster:
        """Return a defensive copy so a failed batch cannot mutate the reference."""

        return deepcopy(self.base_cluster)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.validate()
        base = self.base_cluster
        np.savez_compressed(
            directory / "shared_marker_reference.npz",
            marker_names=self.marker_names.astype(str),
            center=self.center.astype(float),
            scale=self.scale.astype(float),
            normalization_mode=np.asarray([self.normalization_mode], dtype=str),
            n_states=base.n_states.astype(int),
            mu=np.asarray(base.mu, dtype=object),
            variance=np.asarray(base.variance, dtype=object),
            component_weights=np.asarray(base.component_weights, dtype=object),
            marker_status=base.marker_status.astype(str),
            marker_weights=base.marker_weights.astype(float),
            variance_floor=np.asarray([base.variance_floor], dtype=float),
        )
        base.model_selection_metrics.to_csv(
            directory / "shared_marker_reference_metrics.csv",
            index=False,
        )
        summary = base.to_marker_params(
            center=self.center,
            scale=self.scale,
            n_iter=0,
        )
        summary.to_csv(
            directory / "shared_marker_reference_params.csv",
            index=False,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "SharedMarkerReference":
        directory = Path(directory)
        arrays = np.load(
            directory / "shared_marker_reference.npz",
            allow_pickle=True,
        )
        metrics = pd.read_csv(
            directory / "shared_marker_reference_metrics.csv",
            low_memory=False,
        )
        marker_names = arrays["marker_names"].astype(str)
        base = BaseCluster(
            n_states=arrays["n_states"].astype(int),
            mu=[np.asarray(x, dtype=float) for x in arrays["mu"]],
            variance=[np.asarray(x, dtype=float) for x in arrays["variance"]],
            component_weights=[
                np.asarray(x, dtype=float) for x in arrays["component_weights"]
            ],
            marker_status=arrays["marker_status"].astype(str),
            marker_weights=arrays["marker_weights"].astype(float),
            model_selection_metrics=metrics,
            marker_names=marker_names,
            variance_floor=float(arrays["variance_floor"][0]),
        )
        reference = cls(
            marker_names=marker_names,
            center=arrays["center"].astype(float),
            scale=arrays["scale"].astype(float),
            base_cluster=base,
            normalization_mode=(
                str(arrays["normalization_mode"][0])
                if "normalization_mode" in arrays
                else "global"
            ),
        )
        reference.validate()
        return reference


def fit_shared_marker_reference(
    pooled_matrix: np.ndarray,
    marker_names: Iterable[str],
    *,
    random_state: int = 0,
    exclude_technical_markers: bool = True,
    min_component_weight: float = 0.025,
    min_bic_gain: float = 10.0,
    min_separation: float = 1.0,
    min_bhattacharyya: float = 0.08,
    min_mean_posterior_conf: float = 0.70,
    prestandardized: bool = False,
    normalization_mode: str = "global",
) -> SharedMarkerReference:
    """Fit one marker-state scale shared by every selected batch."""

    matrix = np.asarray(pooled_matrix, dtype=np.float64)
    names = np.asarray(list(marker_names), dtype=str)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("pooled matrix must be cells x marker_names")
    if matrix.shape[0] < 2:
        raise ValueError("at least two pooled cells are required")
    if len(set(names)) != len(names):
        raise ValueError("marker names must be unique")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("pooled matrix contains non-finite values")

    jittered = add_gaussian_jitter(
        matrix,
        noise_sd=1e-3,
        random_state=random_state,
        mode="all",
    )
    if prestandardized:
        normalized = jittered
        center = np.zeros(len(names), dtype=float)
        scale = np.ones(len(names), dtype=float)
    else:
        normalized, center, scale = robust_zscore_per_marker(jittered)
    base = BaseCluster.initialize_marker_models(
        normalized,
        marker_names=names,
        random_state=random_state,
        min_component_weight=min_component_weight,
        bic_gain_12=min_bic_gain,
        min_separation=min_separation,
        min_bhattacharyya=min_bhattacharyya,
        min_mean_posterior_conf=min_mean_posterior_conf,
    )
    if exclude_technical_markers:
        excluded = technical_marker_mask(names)
        base.marker_weights[excluded] = 0.0
        base.marker_status[excluded] = "excluded_technical"
        metrics = base.model_selection_metrics.copy()
        metrics.loc[excluded, "phenotype_weight"] = 0.0
        metrics.loc[excluded, "status"] = "excluded_technical"
        metrics["excluded_technical"] = excluded
        base.model_selection_metrics = metrics
    reference = SharedMarkerReference(
        marker_names=names,
        center=center,
        scale=scale,
        base_cluster=base,
        normalization_mode=normalization_mode,
    )
    reference.validate()
    return reference


class FrozenMarkerParamSharedCluster(ParamSharedCluster):
    """ParamSharedCluster that updates phenotype mixtures but not marker states."""

    def m_step(self, X, stats_pack, shrinkage_strength=0.0):
        del X, shrinkage_strength
        stats, theta_counts = stats_pack
        del stats
        counts = self.resp.sum(axis=0)
        prior = self.alpha_pi / max(self.n_components, 1)
        self.pi = np.maximum(counts + prior - 1.0, EPS)
        self.pi /= self.pi.sum()
        for marker_idx in range(self.base_cluster.n_markers):
            start, end = self.base_cluster.state_offsets[marker_idx:marker_idx + 2]
            block = np.maximum(theta_counts[:, start:end], EPS)
            self.theta_flat[:, start:end] = block / block.sum(
                axis=1,
                keepdims=True,
            )


def fit_batch_with_shared_reference(
    normalized_batch_matrix: np.ndarray,
    marker_names: Iterable[str],
    reference: SharedMarkerReference,
    *,
    n_components: int = 40,
    random_state: int = 0,
    min_component_cells: int = 40,
    min_component_fraction: float = 0.0005,
    merge_threshold: float = 0.012,
    redundant_merge_threshold: float = 0.025,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> dict:
    """Fit batch-local phenotype combinations against frozen shared states."""

    names = np.asarray(list(marker_names), dtype=str)
    matrix, calibration = reference.transform_with_diagnostics(
        normalized_batch_matrix,
        names,
    )
    n_cells, n_markers = matrix.shape
    if n_cells == 0:
        raise ValueError("cannot fit an empty batch")
    base = reference.fresh_base_cluster()
    active = int(np.sum(base.marker_weights > 0))
    k = min(max(int(n_components), 1), n_cells)
    if active == 0:
        k = 1
    model = FrozenMarkerParamSharedCluster(
        base,
        n_components=k,
        alpha_theta=1.1,
        alpha_pi=0.08,
        random_state=random_state,
    )
    model.fit(
        matrix,
        max_iter=max_iter,
        tol=tol,
        shrinkage_strength=0.0,
        warmup_iter=5,
        min_component_cells=min(max(int(min_component_cells), 1), n_cells),
        min_component_fraction=min_component_fraction,
        merge_threshold=merge_threshold,
        redundant_merge_threshold=redundant_merge_threshold,
        reweight_markers=False,
    )
    model.e_step(matrix, return_stats=False)
    state_post = model.state_posteriors(matrix)
    high_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    low_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    unimodal_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    state_argmax = np.zeros((n_cells, n_markers), dtype=np.int16)
    for marker_idx, posterior in enumerate(state_post):
        state_argmax[:, marker_idx] = np.argmax(posterior, axis=1)
        if base.n_states[marker_idx] == 1:
            unimodal_prob[:, marker_idx] = 1.0
        else:
            low_prob[:, marker_idx] = posterior[:, 0].astype(np.float32)
            high_prob[:, marker_idx] = posterior[:, -1].astype(np.float32)
    resp = model.resp.astype(np.float32)
    counts = np.bincount(
        resp.argmax(axis=1),
        minlength=model.n_components,
    )
    sorted_resp = np.sort(resp, axis=1)
    margin = (
        sorted_resp[:, -1] - sorted_resp[:, -2]
        if model.n_components > 1
        else sorted_resp[:, -1]
    )
    marker_params = base.to_marker_params(
        center=reference.center,
        scale=reference.scale,
        n_iter=len(model.history),
    )
    marker_params = marker_params.merge(calibration, on="marker", how="left")
    return {
        "high_prob": high_prob,
        "low_prob": low_prob,
        "unimodal_prob": unimodal_prob,
        "empty_prob": np.zeros_like(high_prob),
        "state_argmax": state_argmax,
        "state_posteriors": state_post,
        "phenotype_posterior": resp,
        "posterior_likelihood": resp,
        "phenotype_assignment": np.asarray(model.labels_, dtype=str),
        "phenotype_confidence": resp.max(axis=1).astype(np.float32),
        "phenotype_margin": margin.astype(np.float32),
        "phenotype_counts": counts,
        "phenotype_proportions": model.pi.copy(),
        "theta_flat": model.theta_flat.copy(),
        "pi": model.pi.copy(),
        "marker_state_offsets": base.state_offsets.copy(),
        "marker_n_states": base.n_states.copy(),
        "marker_weights": base.marker_weights.copy(),
        "marker_status": base.marker_status.copy(),
        "model_selection_metrics": base.model_selection_metrics.copy(),
        "marker_params": marker_params,
        "marker_center": reference.center.copy(),
        "marker_scale": reference.scale.copy(),
        "log_likelihood_history": model.history,
        "component_history": model.history,
        "n_iter": len(model.history),
        "model": model,
        "optimizer": model,
    }
