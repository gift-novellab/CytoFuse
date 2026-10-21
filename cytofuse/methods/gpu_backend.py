"""Chunked PyTorch backend for CytoFuse phenotype-mixture EM."""

from __future__ import annotations

import os
import subprocess

import numpy as np


EPS = 1e-12


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "The CUDA backend requires PyTorch. Install a CUDA-enabled torch build."
        ) from exc
    return torch


def _device_with_most_free_memory(torch) -> str:
    """Choose a visible device without creating CUDA contexts on every GPU."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        free_by_physical = {
            int(index.strip()): int(free.strip())
            for index, free in (
                line.split(",", maxsplit=1)
                for line in query.stdout.splitlines()
                if line.strip()
            )
        }
        if visible:
            physical = [int(value.strip()) for value in visible.split(",")]
            visible_index = max(
                range(len(physical)),
                key=lambda index: free_by_physical.get(physical[index], -1),
            )
            return f"cuda:{visible_index}"
        physical_index = max(free_by_physical, key=free_by_physical.get)
        if physical_index < torch.cuda.device_count():
            return f"cuda:{physical_index}"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "cuda:0"


def resolve_backend(backend: str = "auto", device: str | None = None) -> tuple[str, str]:
    """Resolve ``auto/cpu/cuda`` without importing torch for explicit CPU runs."""

    backend = str(backend).lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be one of {'auto', 'cpu', 'cuda'}")
    requested = device or os.environ.get("CYTOFUSE_DEVICE")
    if backend == "cpu" or requested == "cpu":
        return "cpu", "cpu"
    try:
        torch = _load_torch()
    except RuntimeError:
        if backend == "auto":
            return "cpu", "cpu"
        raise
    available = bool(torch.cuda.is_available())
    if backend == "cuda" and not available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if backend == "auto" and not available:
        return "cpu", "cpu"
    if requested:
        return "cuda", str(requested)
    return "cuda", _device_with_most_free_memory(torch)


class TorchEMBackend:
    """Execute the dense E-step and posterior projection on a CUDA device."""

    def __init__(
        self,
        device: str,
        *,
        cell_batch_size: int = 4096,
        marker_block_size: int = 32,
    ) -> None:
        self.torch = _load_torch()
        self.device = self.torch.device(device)
        self.cell_batch_size = max(int(cell_batch_size), 1)
        self.marker_block_size = max(int(marker_block_size), 1)

    def _parameters(self, model):
        torch = self.torch
        base = model.base_cluster
        markers = base.n_markers
        states = base.max_states
        components = model.n_components
        mu = np.zeros((markers, states), dtype=np.float32)
        variance = np.ones((markers, states), dtype=np.float32)
        log_theta = np.full((components, markers, states), -np.inf, dtype=np.float32)
        valid = np.zeros((markers, states), dtype=bool)
        for marker in range(markers):
            count = int(base.n_states[marker])
            start, end = base.state_offsets[marker : marker + 2]
            mu[marker, :count] = base.mu[marker]
            variance[marker, :count] = np.maximum(base.variance[marker], EPS)
            theta = np.maximum(model.theta_flat[:, start:end], EPS)
            log_theta[:, marker, :count] = np.log(theta)
            valid[marker, :count] = True
        return {
            "mu": torch.as_tensor(mu, device=self.device),
            "variance": torch.as_tensor(variance, device=self.device),
            "log_theta": torch.as_tensor(log_theta, device=self.device),
            "valid": torch.as_tensor(valid, device=self.device),
            "weights": torch.as_tensor(
                base.marker_weights.astype(np.float32), device=self.device
            ),
            "log_pi": torch.log(
                torch.as_tensor(
                    np.maximum(model.pi, EPS).astype(np.float32),
                    device=self.device,
                )
            ),
        }

    def _marker_log_probability(self, x, params, marker_index):
        torch = self.torch
        values = x[:, marker_index]
        mu = params["mu"][marker_index]
        variance = params["variance"][marker_index]
        log_probability = (
            -0.5 * torch.log(2.0 * torch.pi * variance)[None, :, :]
            -0.5 * (values[:, :, None] - mu[None, :, :]).square()
            / variance[None, :, :]
        )
        return log_probability.masked_fill(
            ~params["valid"][marker_index][None, :, :], -torch.inf
        )

    def _component_responsibility(self, x, params, active_markers):
        torch = self.torch
        log_resp = params["log_pi"][None, :].expand(x.shape[0], -1).clone()
        for start in range(0, len(active_markers), self.marker_block_size):
            marker_index = active_markers[start : start + self.marker_block_size]
            log_probability = self._marker_log_probability(x, params, marker_index)
            joint = (
                log_probability[:, None, :, :]
                + params["log_theta"][:, marker_index, :][None, :, :, :]
            )
            denominator = torch.logsumexp(joint, dim=-1)
            log_resp += (
                denominator * params["weights"][marker_index][None, None, :]
            ).sum(dim=-1)
        normalizer = torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp - normalizer), normalizer.sum()

    def e_step(self, model, X, *, return_stats: bool):
        """Run one E-step, reducing the cell chunk automatically after OOM."""

        batch_size = self.cell_batch_size
        while True:
            try:
                result = self._e_step_once(
                    model, X, return_stats=return_stats, batch_size=batch_size
                )
                self.cell_batch_size = batch_size
                return result
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if batch_size <= 256:
                    raise
                batch_size = max(batch_size // 2, 256)

    def _e_step_once(self, model, X, *, return_stats: bool, batch_size: int):
        torch = self.torch
        values = np.asarray(X, dtype=np.float32)
        n_cells, n_markers = values.shape
        params = self._parameters(model)
        active = np.flatnonzero(model.base_cluster.marker_weights > 0).tolist()
        resp = np.empty((n_cells, model.n_components), dtype=np.float32)
        log_likelihood = 0.0
        max_states = model.base_cluster.max_states
        if return_stats:
            theta_counts = torch.zeros(
                (model.n_components, n_markers, max_states),
                dtype=torch.float32,
                device=self.device,
            )
            resp_sum = torch.zeros(
                (n_markers, max_states), dtype=torch.float32, device=self.device
            )
            x_sum = torch.zeros_like(resp_sum)
            x2_sum = torch.zeros_like(resp_sum)
        with torch.inference_mode():
            for cell_start in range(0, n_cells, batch_size):
                cell_end = min(cell_start + batch_size, n_cells)
                x = torch.as_tensor(
                    values[cell_start:cell_end], device=self.device
                )
                chunk_resp, chunk_ll = self._component_responsibility(
                    x, params, active
                )
                resp[cell_start:cell_end] = chunk_resp.cpu().numpy()
                log_likelihood += float(chunk_ll.item())
                if not return_stats:
                    continue
                for marker_start in range(
                    0, len(active), self.marker_block_size
                ):
                    marker_index = active[
                        marker_start : marker_start + self.marker_block_size
                    ]
                    log_probability = self._marker_log_probability(
                        x, params, marker_index
                    )
                    joint = (
                        log_probability[:, None, :, :]
                        + params["log_theta"][:, marker_index, :][None, :, :, :]
                    )
                    state_given_component = torch.softmax(joint, dim=-1)
                    weighted = (
                        chunk_resp[:, :, None, None] * state_given_component
                    )
                    theta_counts[:, marker_index, :] += weighted.sum(dim=0)
                    total_state = weighted.sum(dim=1)
                    resp_sum[marker_index, :] += total_state.sum(dim=0)
                    marker_values = x[:, marker_index, None]
                    x_sum[marker_index, :] += (
                        total_state * marker_values
                    ).sum(dim=0)
                    x2_sum[marker_index, :] += (
                        total_state * marker_values.square()
                    ).sum(dim=0)

        model.resp = resp.astype(float, copy=False)
        model.log_likelihood = float(log_likelihood)
        model.labels_ = np.argmax(resp, axis=1).astype(str)
        if not return_stats:
            return model.resp, model.log_likelihood, None

        prior = max(model.alpha_theta - 1.0, 0.0)
        theta_flat = np.zeros_like(model.theta_flat)
        marker_resp_sum = []
        marker_x_sum = []
        marker_x2_sum = []
        theta_cpu = theta_counts.cpu().numpy()
        resp_cpu = resp_sum.cpu().numpy()
        x_cpu = x_sum.cpu().numpy()
        x2_cpu = x2_sum.cpu().numpy()
        active_set = set(active)
        for marker, count in enumerate(model.base_cluster.n_states):
            count = int(count)
            start, end = model.base_cluster.state_offsets[marker : marker + 2]
            if marker in active_set:
                theta_flat[:, start:end] = theta_cpu[:, marker, :count] + prior
                marker_resp_sum.append(resp_cpu[marker, :count].astype(float))
                marker_x_sum.append(x_cpu[marker, :count].astype(float))
                marker_x2_sum.append(x2_cpu[marker, :count].astype(float))
            else:
                theta_flat[:, start:end] = 1.0 / count
                marker_resp_sum.append(np.zeros(count, dtype=float))
                marker_x_sum.append(np.zeros(count, dtype=float))
                marker_x2_sum.append(np.zeros(count, dtype=float))
        from cytofuse.methods.paramshared import MarkerStateStats

        stats = MarkerStateStats(marker_resp_sum, marker_x_sum, marker_x2_sum)
        return model.resp, model.log_likelihood, (stats, theta_flat)

    def state_posteriors(self, model, X):
        """Project phenotype responsibilities back to marker-state posteriors."""

        torch = self.torch
        values = np.asarray(X, dtype=np.float32)
        params = self._parameters(model)
        output = np.zeros(
            (len(values), model.base_cluster.n_markers, model.base_cluster.max_states),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for cell_start in range(0, len(values), self.cell_batch_size):
                cell_end = min(cell_start + self.cell_batch_size, len(values))
                x = torch.as_tensor(values[cell_start:cell_end], device=self.device)
                responsibility = torch.as_tensor(
                    model.resp[cell_start:cell_end].astype(np.float32),
                    device=self.device,
                )
                for marker_start in range(
                    0, model.base_cluster.n_markers, self.marker_block_size
                ):
                    marker_index = list(
                        range(
                            marker_start,
                            min(
                                marker_start + self.marker_block_size,
                                model.base_cluster.n_markers,
                            ),
                        )
                    )
                    log_probability = self._marker_log_probability(
                        x, params, marker_index
                    )
                    joint = (
                        log_probability[:, None, :, :]
                        + params["log_theta"][:, marker_index, :][None, :, :, :]
                    )
                    state_given_component = torch.softmax(joint, dim=-1)
                    posterior = (
                        responsibility[:, :, None, None] * state_given_component
                    ).sum(dim=1)
                    output[cell_start:cell_end, marker_index, :] = (
                        posterior.cpu().numpy()
                    )
        return [
            output[:, marker, : int(count)]
            for marker, count in enumerate(model.base_cluster.n_states)
        ]
