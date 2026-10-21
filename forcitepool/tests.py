"""Small synthetic tests for the standalone taxonomy rules."""

from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import pandas as pd

from forcitepool.evaluate import safe_filename_token
from forcitepool.run import (
    make_batch_file_ids,
    normalize_proteins,
    protein_matrix_and_names,
    resolve_batches,
)
from forcitepool.tree import (
    StateThresholds,
    align_clusters,
    build_cluster_marker_states,
    build_taxonomy,
)
from cytofuse.methods.paramshared import MarkerModelSelector, fit_shared_marker_gmm


def state_table(records):
    rows = []
    for batch, cluster, states in records:
        for marker, state in states.items():
            rows.append(
                {
                    "batch": batch,
                    "cluster": cluster,
                    "local_node_id": f"{batch}|{cluster}",
                    "n_cells": 100,
                    "marker": marker,
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


class AlignmentTests(unittest.TestCase):
    def test_two_weak_cross_batch_conflicts_can_align(self):
        shared = {f"shared_{index}": "+" for index in range(20)}
        table = state_table(
            [
                ("b1", "T", {**shared, "m4": "-", "m5": "+"}),
                ("b2", "T", {**shared, "m4": "+", "m5": "-"}),
            ]
        )
        table["call_confidence"] = np.where(
            table["marker"].isin(["m4", "m5"]), 0.55, 0.95
        )
        result = align_clusters(
            table,
            min_shared_observed=3,
            min_alignment_agreement=0.75,
            min_configuration_coverage=0.50,
        )
        self.assertEqual(result["membership"]["aligned_node_id"].nunique(), 1)

    def test_three_weak_cross_batch_conflicts_do_not_align(self):
        table = state_table(
            [
                ("b1", "T", {"m1": "+", "m2": "+", "m3": "+", "m4": "-", "m5": "-", "m6": "-"}),
                ("b2", "T", {"m1": "+", "m2": "+", "m3": "+", "m4": "+", "m5": "+", "m6": "+"}),
            ]
        )
        table["call_confidence"] = np.where(
            table["marker"].isin(["m4", "m5", "m6"]), 0.55, 0.95
        )
        result = align_clusters(
            table,
            min_shared_observed=3,
            min_alignment_agreement=0.50,
            min_configuration_coverage=0.50,
        )
        self.assertEqual(result["membership"]["aligned_node_id"].nunique(), 2)

    def test_transitive_conflict_is_not_joined(self):
        table = state_table(
            [
                ("a", "1", {"m1": "+", "m2": "?"}),
                ("b", "1", {"m1": "?", "m2": "+"}),
                ("c", "1", {"m1": "-", "m2": "+"}),
            ]
        )
        result = align_clusters(table, min_shared_observed=1)
        groups = result["membership"].groupby("aligned_node_id")[
            "local_node_id"
        ].apply(set)
        self.assertFalse(any({"a|1", "c|1"}.issubset(group) for group in groups))

    def test_same_batch_clusters_never_align(self):
        table = state_table(
            [
                ("a", "1", {"m1": "+", "m2": "-"}),
                ("a", "2", {"m1": "+", "m2": "-"}),
            ]
        )
        result = align_clusters(table, min_shared_observed=1)
        self.assertEqual(result["membership"]["aligned_node_id"].nunique(), 2)

    def test_two_batch_prototype_absorbs_one_unknown_batch(self):
        table = state_table(
            [
                (
                    "b1",
                    "T",
                    {"CD3": "+", "CD8": "+", "CD19": "-", "CD14": "-", "CCR7": "+"},
                ),
                (
                    "b3",
                    "T",
                    {"CD3": "+", "CD8": "+", "CD19": "-", "CD14": "-", "CCR7": "+"},
                ),
                (
                    "b2",
                    "T",
                    {"CD3": "+", "CD8": "?", "CD19": "-", "CD14": "-", "CCR7": "+"},
                ),
            ]
        )
        result = align_clusters(
            table,
            min_shared_observed=3,
            use_weighted_scores=False,
        )
        groups = result["membership"].groupby("aligned_node_id")[
            "local_node_id"
        ].apply(set)
        self.assertTrue(
            any({"b1|T", "b2|T", "b3|T"} == group for group in groups)
        )

    def test_asymmetric_coarse_configuration_is_not_forced_equivalent(self):
        table = state_table(
            [
                (
                    "b1",
                    "coarse",
                    {"CD3": "+", "CD4": "+", "CCR7": "?", "CD45RA": "?", "CD27": "?"},
                ),
                (
                    "b2",
                    "fine",
                    {"CD3": "+", "CD4": "+", "CCR7": "+", "CD45RA": "-", "CD27": "+"},
                ),
            ]
        )
        result = align_clusters(
            table,
            min_shared_observed=2,
            min_configuration_coverage=0.50,
            use_weighted_scores=False,
        )
        self.assertEqual(result["membership"]["aligned_node_id"].nunique(), 2)

    def test_repeated_high_confidence_conflicts_stay_separate(self):
        table = state_table(
            [
                (
                    "b1",
                    "T",
                    {"CD3": "+", "CD8": "+", "CCR7": "+", "CD19": "-", "CD14": "-"},
                ),
                (
                    "b3",
                    "T",
                    {"CD3": "+", "CD8": "+", "CCR7": "+", "CD19": "-", "CD14": "-"},
                ),
                (
                    "b2",
                    "other",
                    {"CD3": "+", "CD8": "-", "CCR7": "-", "CD19": "-", "CD14": "-"},
                ),
            ]
        )
        result = align_clusters(
            table,
            min_shared_observed=3,
            use_weighted_scores=False,
        )
        groups = result["membership"].groupby("aligned_node_id")[
            "local_node_id"
        ].apply(set)
        self.assertFalse(
            any({"b1|T", "b2|other", "b3|T"} == group for group in groups)
        )

    def test_many_batches_form_pure_components_without_duplicate_batches(self):
        records = []
        signatures = [
            {"m1": "+", "m2": "+", "m3": "-", "m4": "-"},
            {"m1": "+", "m2": "-", "m3": "+", "m4": "-"},
            {"m1": "-", "m2": "+", "m3": "-", "m4": "+"},
            {"m1": "-", "m2": "-", "m3": "+", "m4": "+"},
        ]
        for batch_index in range(40):
            for type_index, signature in enumerate(signatures):
                states = dict(signature)
                if (batch_index + type_index) % 11 == 0:
                    states["m4"] = "?"
                records.append(
                    (f"b{batch_index:02d}", f"t{type_index}", states)
                )
        result = align_clusters(
            state_table(records),
            min_shared_observed=3,
            use_weighted_scores=False,
        )
        membership = result["membership"].copy()
        membership["truth"] = membership["cluster"].str.removeprefix("t")
        self.assertEqual(membership["aligned_node_id"].nunique(), 4)
        self.assertTrue(
            membership.groupby("aligned_node_id")["truth"].nunique().eq(1).all()
        )
        self.assertTrue(
            membership.groupby("aligned_node_id")["batch"].nunique().eq(40).all()
        )


class WorkflowInputTests(unittest.TestCase):
    def test_all_batches_are_selected_in_input_order_by_default(self):
        values = pd.Series(["donor B", "donor A", "donor B", "donor C"])
        self.assertEqual(
            resolve_batches(values, None),
            ["donor B", "donor A", "donor C"],
        )

    def test_explicit_batch_subset_is_preserved(self):
        values = pd.Series(["a", "b", "c"])
        self.assertEqual(resolve_batches(values, ["c", "a"]), ["c", "a"])
        with self.assertRaises(ValueError):
            resolve_batches(values, ["missing"])

    def test_arbitrary_batch_labels_get_unique_safe_file_ids(self):
        mapping = make_batch_file_ids(["donor/1", "donor 1", "safe.batch"])
        self.assertEqual(mapping["safe.batch"], "safe.batch")
        self.assertNotEqual(mapping["donor/1"], mapping["donor 1"])
        self.assertNotIn("/", mapping["donor/1"])
        self.assertNotIn(" ", mapping["donor 1"])

    def test_protein_data_can_come_from_x_layer_or_obsm(self):
        adata = ad.AnnData(
            X=np.ones((3, 2)),
            layers={"adt_layer": np.full((3, 2), 2.0)},
        )
        adata.var_names = ["x1", "x2"]
        matrix, names = protein_matrix_and_names(adata)
        np.testing.assert_array_equal(matrix, np.ones((3, 2)))
        self.assertEqual(names.tolist(), ["x1", "x2"])

        matrix, names = protein_matrix_and_names(
            adata,
            protein_layer="adt_layer",
        )
        np.testing.assert_array_equal(matrix, np.full((3, 2), 2.0))
        self.assertEqual(names.tolist(), ["x1", "x2"])

        adata.obsm["adt"] = pd.DataFrame(
            np.full((3, 2), 3.0),
            index=adata.obs_names,
            columns=["CD3", "CD19"],
        )
        matrix, names = protein_matrix_and_names(
            adata,
            protein_obsm_key="adt",
        )
        np.testing.assert_array_equal(matrix, np.full((3, 2), 3.0))
        self.assertEqual(names.tolist(), ["CD3", "CD19"])

    def test_preprocessed_negative_values_require_no_normalization(self):
        matrix = np.array([[-1.0, 1.0], [0.0, 2.0]])
        np.testing.assert_array_equal(
            normalize_proteins(matrix, "none"),
            matrix,
        )
        with self.assertRaises(ValueError):
            normalize_proteins(matrix, "clr")

    def test_annotation_names_are_safe_for_output_filenames(self):
        token = safe_filename_token("cell/type 中文")
        self.assertNotIn("/", token)
        self.assertNotIn(" ", token)


class TaxonomyTests(unittest.TestCase):
    @staticmethod
    def membership(nodes):
        return pd.DataFrame(
            {
                "aligned_node_id": nodes,
                "local_node_id": [f"b{i}|1" for i in range(len(nodes))],
                "batch": [f"b{i}" for i in range(len(nodes))],
                "cluster": ["1"] * len(nodes),
                "n_cells": [100] * len(nodes),
            }
        )

    def test_positive_then_negative_lexicographic_priority(self):
        membership = pd.DataFrame(
            {
                "aligned_node_id": ["A", "B", "C"],
                "local_node_id": ["a|1", "b|1", "c|1"],
                "batch": ["a", "b", "c"],
                "cluster": ["1", "1", "1"],
                "n_cells": [100, 100, 100],
            }
        )
        states = pd.DataFrame(
            [
                {"aligned_node_id": "A", "marker": "p1", "state": "+"},
                {"aligned_node_id": "A", "marker": "p2", "state": "+"},
                {"aligned_node_id": "A", "marker": "n", "state": "?"},
                {"aligned_node_id": "B", "marker": "p1", "state": "+"},
                {"aligned_node_id": "B", "marker": "p2", "state": "+"},
                {"aligned_node_id": "B", "marker": "n", "state": "-"},
                {"aligned_node_id": "C", "marker": "p1", "state": "?"},
                {"aligned_node_id": "C", "marker": "p2", "state": "?"},
                {"aligned_node_id": "C", "marker": "n", "state": "-"},
            ]
        )
        result = build_taxonomy(membership, states)
        first = result["merges"].sort_values("round").iloc[0]
        self.assertEqual({first["child_a"], first["child_b"]}, {"A", "B"})

    def test_tied_candidates_do_not_merge(self):
        membership = pd.DataFrame(
            {
                "aligned_node_id": ["A", "B", "C"],
                "local_node_id": ["a|1", "b|1", "c|1"],
                "batch": ["a", "b", "c"],
                "cluster": ["1", "1", "1"],
                "n_cells": [100, 100, 100],
            }
        )
        states = pd.DataFrame(
            [
                {"aligned_node_id": node, "marker": "m", "state": "+"}
                for node in ["A", "B", "C"]
            ]
        )
        result = build_taxonomy(membership, states)
        self.assertTrue(result["merges"].empty)
        self.assertEqual(len(result["forest"]["roots"]), 3)

    def test_zero_shared_evidence_stays_a_forest(self):
        membership = pd.DataFrame(
            {
                "aligned_node_id": ["A", "B"],
                "local_node_id": ["a|1", "b|1"],
                "batch": ["a", "b"],
                "cluster": ["1", "1"],
                "n_cells": [100, 100],
            }
        )
        states = pd.DataFrame(
            [
                {"aligned_node_id": "A", "marker": "m1", "state": "+"},
                {"aligned_node_id": "A", "marker": "m2", "state": "?"},
                {"aligned_node_id": "B", "marker": "m1", "state": "?"},
                {"aligned_node_id": "B", "marker": "m2", "state": "-"},
            ]
        )
        result = build_taxonomy(membership, states)
        self.assertTrue(result["merges"].empty)
        self.assertEqual(set(result["forest"]["roots"]), {"A", "B"})
        self.assertFalse(result["pair_audit"]["eligible"].any())

    def test_conflict_gate_rejects_pair_without_reordering_scores(self):
        membership = pd.DataFrame(
            {
                "aligned_node_id": ["A", "B"],
                "local_node_id": ["a|1", "b|1"],
                "batch": ["a", "b"],
                "cluster": ["1", "1"],
                "n_cells": [100, 100],
            }
        )
        states = pd.DataFrame(
            [
                {"aligned_node_id": "A", "marker": "same", "state": "+"},
                {"aligned_node_id": "A", "marker": "conflict", "state": "+"},
                {"aligned_node_id": "B", "marker": "same", "state": "+"},
                {"aligned_node_id": "B", "marker": "conflict", "state": "-"},
            ]
        )
        rejected = build_taxonomy(
            membership,
            states,
            max_tree_conflicts=0,
        )
        accepted = build_taxonomy(
            membership,
            states,
            max_tree_conflicts=1,
        )
        self.assertTrue(rejected["merges"].empty)
        self.assertEqual(len(accepted["merges"]), 1)

    def test_confidence_breaks_an_integer_count_tie(self):
        membership = self.membership(["A", "B", "C"])
        states = pd.DataFrame(
            [
                {"aligned_node_id": "A", "marker": "p1", "state": "+", "confidence": 0.9},
                {"aligned_node_id": "A", "marker": "p2", "state": "+", "confidence": 0.9},
                {"aligned_node_id": "B", "marker": "p1", "state": "+", "confidence": 0.9},
                {"aligned_node_id": "B", "marker": "p2", "state": "?", "confidence": 0.0},
                {"aligned_node_id": "C", "marker": "p1", "state": "?", "confidence": 0.0},
                {"aligned_node_id": "C", "marker": "p2", "state": "+", "confidence": 0.4},
            ]
        )
        result = build_taxonomy(membership, states)
        first = result["merges"].iloc[0]
        self.assertEqual({first["child_a"], first["child_b"]}, {"A", "B"})
        self.assertGreater(
            first["weighted_common_positive"],
            0.0,
        )

    def test_tolerance_rejects_a_numerically_tiny_advantage(self):
        membership = self.membership(["A", "B", "C"])
        states = pd.DataFrame(
            [
                {"aligned_node_id": "A", "marker": "p1", "state": "+", "confidence": 0.9},
                {"aligned_node_id": "A", "marker": "p2", "state": "+", "confidence": 0.9},
                {"aligned_node_id": "B", "marker": "p1", "state": "+", "confidence": 0.90},
                {"aligned_node_id": "B", "marker": "p2", "state": "?", "confidence": 0.0},
                {"aligned_node_id": "C", "marker": "p1", "state": "?", "confidence": 0.0},
                {"aligned_node_id": "C", "marker": "p2", "state": "+", "confidence": 0.88},
            ]
        )
        result = build_taxonomy(
            membership,
            states,
            score_absolute_tolerance=0.05,
        )
        self.assertTrue(result["merges"].empty)

    def test_common_states_are_downweighted_relative_to_specific_states(self):
        membership = self.membership(["A", "B", "C", "D"])
        rows = []
        for node in ["A", "B", "C", "D"]:
            rows.append(
                {
                    "aligned_node_id": node,
                    "marker": "common_negative",
                    "state": "-",
                    "confidence": 1.0,
                }
            )
            rows.append(
                {
                    "aligned_node_id": node,
                    "marker": "specific_positive",
                    "state": "+" if node in {"A", "B"} else "?",
                    "confidence": 1.0 if node in {"A", "B"} else 0.0,
                }
            )
        result = build_taxonomy(membership, pd.DataFrame(rows))
        audit = result["pair_audit"]
        pair = audit[
            (audit["node_a"] == "A")
            & (audit["node_b"] == "B")
            & (audit["round"] == 1)
        ].iloc[0]
        self.assertGreater(
            pair["weighted_common_positive"],
            pair["weighted_common_negative"],
        )


class StateTableTests(unittest.TestCase):
    def test_marker_cap_only_limits_explanatory_ranking(self):
        labels = np.repeat(["0", "1"], 50)
        markers = np.array([f"m{i}" for i in range(6)])
        low = np.vstack(
            [
                np.full((50, 6), 0.05),
                np.full((50, 6), 0.95),
            ]
        )
        high = 1.0 - low
        params = pd.DataFrame(
            {
                "n_states": [2] * 6,
                "status": ["valid_bimodal"] * 6,
                "phenotype_weight": [1.0] * 6,
            }
        )
        table = build_cluster_marker_states(
            batch_id="b",
            cluster_labels=labels,
            marker_names=markers,
            low_prob=low,
            high_prob=high,
            marker_params=params,
            thresholds=StateThresholds(
                min_marker_weight=0,
                min_state_confidence=0,
                min_state_separation=0,
                min_cluster_fraction=0,
                min_marker_effect=0,
                min_cell_state_probability=0,
                max_markers_per_cluster=3,
                max_markers_per_state=3,
            ),
        )
        called = table.groupby("local_node_id")["state"].apply(
            lambda x: int(x.ne("?").sum())
        )
        ranked = table.groupby("local_node_id")["is_retained"].sum()
        self.assertTrue((called == 6).all())
        self.assertTrue((ranked <= 3).all())
        callable_rows = table[table["state"] != "?"]
        self.assertTrue(callable_rows["call_confidence"].between(0.0, 1.0).all())
        self.assertTrue(
            (
                callable_rows["cluster_state_fraction_lower"]
                <= callable_rows["cluster_state_fraction"]
            ).all()
        )

    def test_identity_state_does_not_require_batch_local_effect(self):
        labels = np.repeat(["0", "1"], 50)
        low = np.full((100, 1), 0.02)
        high = 1.0 - low
        params = pd.DataFrame(
            {
                "n_states": [2],
                "status": ["valid_bimodal"],
                "phenotype_weight": [1.0],
            }
        )
        table = build_cluster_marker_states(
            batch_id="b",
            cluster_labels=labels,
            marker_names=["CD3"],
            low_prob=low,
            high_prob=high,
            marker_params=params,
            thresholds=StateThresholds(
                min_marker_weight=0,
                min_state_confidence=0,
                min_state_separation=0,
                min_cluster_fraction=0,
                min_marker_effect=0.10,
                min_cell_state_probability=0,
            ),
        )
        self.assertTrue(table["state"].eq("+").all())
        self.assertTrue(np.allclose(table["marker_effect"], 0.0))
        self.assertFalse(table["is_retained"].any())


class MarkerWeightTests(unittest.TestCase):
    @staticmethod
    def overlapping_mixture():
        rng = np.random.default_rng(7)
        return np.concatenate(
            [
                rng.normal(-1.0, 0.8, 1000),
                rng.normal(1.0, 0.8, 1000),
            ]
        )

    def test_weak_two_state_marker_can_cluster_without_being_callable(self):
        selector = MarkerModelSelector(
            random_state=0,
            min_bhattacharyya=10.0,
        )
        selected, metrics = selector.select_marker_state_count(
            self.overlapping_mixture()
        )
        self.assertEqual(selected.n_states, 2)
        self.assertEqual(metrics["status"], "weak_bimodal")
        self.assertGreater(metrics["cluster_weight"], 0.0)
        self.assertEqual(metrics["configuration_weight"], 0.0)

    def test_reliable_two_state_marker_has_both_weights(self):
        selector = MarkerModelSelector(
            random_state=0,
            min_bhattacharyya=0.01,
        )
        selected, metrics = selector.select_marker_state_count(
            self.overlapping_mixture()
        )
        self.assertEqual(selected.n_states, 2)
        self.assertEqual(metrics["status"], "valid_bimodal")
        self.assertGreater(metrics["cluster_weight"], 0.0)
        self.assertGreater(metrics["configuration_weight"], 0.0)


class StructuralEMTests(unittest.TestCase):
    def test_leiden_structural_em_recovers_marker_state_combinations(self):
        rng = np.random.default_rng(7)
        truth = rng.integers(0, 4, 600)
        values = np.column_stack(
            [
                rng.normal(np.where((truth & 1) > 0, 2, -2), 0.55),
                rng.normal(np.where((truth & 2) > 0, 2, -2), 0.55),
                rng.normal(0, 1, len(truth)),
            ]
        )
        result = fit_shared_marker_gmm(
            values,
            marker_names=["A", "B", "noise"],
            K_init=2,
            random_state=3,
            max_iter=40,
            min_component_cells=15,
            discrimination_reweight=False,
        )
        labels = np.asarray(result["phenotype_assignment"])
        self.assertEqual(result["initialization_method"], "leiden")
        self.assertEqual(np.unique(labels).size, 4)
        self.assertEqual(result["phenotype_posterior"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
