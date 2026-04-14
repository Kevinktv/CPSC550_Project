from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None
    DataLoader = None
    TensorDataset = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import benchmark_utils  # noqa: E402
import data_utils  # noqa: E402
import model_utils  # noqa: E402
import notebook_workflows  # noqa: E402


def _tiny_model_factory(num_classes: int, dataset: str | None = None) -> torch.nn.Module:
    del dataset
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, num_classes),
    )


def _tiny_conv_model_factory(num_classes: int, dataset: str | None = None) -> torch.nn.Module:
    del dataset
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(4, num_classes),
    )


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class EfficiencySelectionTests(unittest.TestCase):
    def test_select_efficiency_variant_picks_first_budget_passing_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_dir = root / "baseline_retrain"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for seed, runtime in enumerate([10.0, 10.0, 10.0]):
                (reference_dir / f"seed_{seed}.json").write_text(
                    json.dumps({"runtime_seconds": runtime}),
                    encoding="utf-8",
                )
            trial_results = {
                "default": {"runtime_seconds": 3.5, "best_val_accuracy": 0.9},
                "faster": {"runtime_seconds": 1.8, "best_val_accuracy": 0.85},
                "fastest": {"runtime_seconds": 1.2, "best_val_accuracy": 0.8},
            }
            selected = notebook_workflows._select_efficiency_variant(
                algorithm_name="SCRUB",
                output_family_name="SCRUB",
                candidate_variants=[
                    ("default", {"epochs": 10}),
                    ("faster", {"epochs": 6}),
                    ("fastest", {"epochs": 4}),
                ],
                reference_family_dir=reference_dir,
                efficiency_ratio=0.2,
                trial_runner=lambda variant_name, _variant_config: dict(trial_results[variant_name]),
            )
            self.assertEqual(selected["selected_variant"], "faster")
            self.assertTrue(selected["passed_budget"])
            self.assertEqual(len(selected["trials"]), 2)

    def test_delete_efficiency_variants_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_dir = root / "baseline_retrain"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for seed, runtime in enumerate([12.0, 12.0, 12.0]):
                (reference_dir / f"seed_{seed}.json").write_text(
                    json.dumps({"runtime_seconds": runtime}),
                    encoding="utf-8",
                )
            variants = notebook_workflows._build_delete_efficiency_variants(
                "cifar10",
                notebook_workflows.DELETE_UNLEARNING_PROFILES["cifar10"],
            )
            self.assertEqual([variant_name for variant_name, _ in variants], ["cifar10", "cifar10_epochs_6", "cifar10_epochs_4", "cifar10_epochs_3", "cifar10_epochs_2", "cifar10_epochs_1"])
            trial_results = {
                "cifar10": {"runtime_seconds": 4.0, "best_val_accuracy": 0.88},
                "cifar10_epochs_6": {"runtime_seconds": 3.0, "best_val_accuracy": 0.86},
                "cifar10_epochs_4": {"runtime_seconds": 2.3, "best_val_accuracy": 0.85},
                "cifar10_epochs_3": {"runtime_seconds": 1.8, "best_val_accuracy": 0.82},
                "cifar10_epochs_2": {"runtime_seconds": 1.5, "best_val_accuracy": 0.8},
                "cifar10_epochs_1": {"runtime_seconds": 1.2, "best_val_accuracy": 0.78},
            }
            selected = notebook_workflows._select_efficiency_variant(
                algorithm_name="DELETE",
                output_family_name="DELETE",
                candidate_variants=variants,
                reference_family_dir=reference_dir,
                efficiency_ratio=0.2,
                trial_runner=lambda variant_name, _variant_config: dict(trial_results[variant_name]),
            )
            self.assertEqual(selected["selected_variant"], "cifar10_epochs_4")
            self.assertTrue(selected["passed_budget"])

    def test_msg_efficiency_variants_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_dir = root / "baseline_retrain"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for seed, runtime in enumerate([12.0, 12.0, 12.0]):
                (reference_dir / f"seed_{seed}.json").write_text(
                    json.dumps({"runtime_seconds": runtime}),
                    encoding="utf-8",
                )
            variants = notebook_workflows._build_msg_efficiency_variants(
                "cifar10",
                notebook_workflows.MSG_UNLEARNING_PROFILES["cifar10"],
            )
            self.assertEqual(
                [variant_name for variant_name, _ in variants],
                ["cifar10", "cifar10_epochs_4", "cifar10_epochs_3", "cifar10_epochs_2", "cifar10_epochs_1"],
            )
            trial_results = {
                "cifar10": {"runtime_seconds": 4.0, "best_val_accuracy": 0.88},
                "cifar10_epochs_4": {"runtime_seconds": 3.1, "best_val_accuracy": 0.86},
                "cifar10_epochs_3": {"runtime_seconds": 2.1, "best_val_accuracy": 0.84},
                "cifar10_epochs_2": {"runtime_seconds": 1.6, "best_val_accuracy": 0.8},
                "cifar10_epochs_1": {"runtime_seconds": 1.0, "best_val_accuracy": 0.77},
            }
            selected = notebook_workflows._select_efficiency_variant(
                algorithm_name="MSG",
                output_family_name="MSG",
                candidate_variants=variants,
                reference_family_dir=reference_dir,
                efficiency_ratio=0.2,
                trial_runner=lambda variant_name, _variant_config: dict(trial_results[variant_name]),
            )
            self.assertEqual(selected["selected_variant"], "cifar10_epochs_3")
            self.assertTrue(selected["passed_budget"])

    def test_ct_efficiency_variants_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_dir = root / "baseline_retrain"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for seed, runtime in enumerate([8.0, 8.0, 8.0]):
                (reference_dir / f"seed_{seed}.json").write_text(
                    json.dumps({"runtime_seconds": runtime}),
                    encoding="utf-8",
                )
            variants = notebook_workflows._build_ct_efficiency_variants(
                "cifar10",
                notebook_workflows.CT_UNLEARNING_PROFILES["cifar10"],
            )
            self.assertEqual(
                [variant_name for variant_name, _ in variants],
                ["cifar10", "cifar10_epochs_2", "cifar10_epochs_1"],
            )
            trial_results = {
                "cifar10": {"runtime_seconds": 2.4, "best_val_accuracy": 0.86},
                "cifar10_epochs_2": {"runtime_seconds": 1.5, "best_val_accuracy": 0.84},
                "cifar10_epochs_1": {"runtime_seconds": 1.0, "best_val_accuracy": 0.79},
            }
            selected = notebook_workflows._select_efficiency_variant(
                algorithm_name="CT",
                output_family_name="CT",
                candidate_variants=variants,
                reference_family_dir=reference_dir,
                efficiency_ratio=0.2,
                trial_runner=lambda variant_name, _variant_config: dict(trial_results[variant_name]),
            )
            self.assertEqual(selected["selected_variant"], "cifar10_epochs_2")
            self.assertTrue(selected["passed_budget"])


@unittest.skipIf(torch is None, "torch is required for workflow smoke tests")
class NotebookWorkflowSmokeTests(unittest.TestCase):
    def _make_record(self, *, sample_id: str, split_name: str, class_index: int) -> data_utils.SampleRecord:
        return data_utils.SampleRecord(
            sample_id=sample_id,
            dataset="cifar10",
            source_partition=split_name,
            label_field="label_id",
            label_id=str(class_index),
            label_name=f"class_{class_index}",
            raw_index=int(sample_id.split("_")[-1]),
            relative_path=f"{sample_id}.png",
            class_index=class_index,
        )

    def _build_bundle(self, root: Path, *, task_id: str = "forget_smoke") -> data_utils.DataBundle:
        generator = torch.Generator().manual_seed(7)
        retain_inputs = torch.randn(4, 3, 4, 4, generator=generator)
        retain_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        forget_inputs = torch.randn(2, 3, 4, 4, generator=generator)
        forget_targets = torch.tensor([0, 1], dtype=torch.long)
        val_inputs = torch.randn(2, 3, 4, 4, generator=generator)
        val_targets = torch.tensor([0, 1], dtype=torch.long)
        test_inputs = torch.randn(2, 3, 4, 4, generator=generator)
        test_targets = torch.tensor([1, 0], dtype=torch.long)
        train_inputs = torch.cat([retain_inputs, forget_inputs], dim=0)
        train_targets = torch.cat([retain_targets, forget_targets], dim=0)

        train_records = [
            self._make_record(sample_id=f"train_{index}", split_name="train", class_index=int(label))
            for index, label in enumerate(train_targets.tolist())
        ]
        retain_records = [
            self._make_record(sample_id=f"retain_{index}", split_name="retrain", class_index=int(label))
            for index, label in enumerate(retain_targets.tolist())
        ]
        forget_records = [
            self._make_record(sample_id=f"forget_{index}", split_name="forget", class_index=int(label))
            for index, label in enumerate(forget_targets.tolist())
        ]
        val_records = [
            self._make_record(sample_id=f"val_{index}", split_name="val", class_index=int(label))
            for index, label in enumerate(val_targets.tolist())
        ]
        test_records = [
            self._make_record(sample_id=f"test_{index}", split_name="test", class_index=int(label))
            for index, label in enumerate(test_targets.tolist())
        ]
        all_records = train_records + retain_records + forget_records + val_records + test_records
        context = data_utils.ManifestContext(
            dataset="cifar10",
            task_id=task_id,
            label_field="label_id",
            data_root=root,
            task_manifest_path=root / "task_manifest.json",
            samples_csv_path=root / "samples.csv",
            label_to_index={"0": 0, "1": 1},
            index_to_label={0: "0", 1: "1"},
            class_names=["class_0", "class_1"],
            sample_lookup={record.sample_id: record for record in all_records},
            splits={
                "train": train_records,
                "retrain": retain_records,
                "forget": forget_records,
                "val": val_records,
                "test": test_records,
            },
            extra_eval_sets={},
        )
        loaders = {
            "train": DataLoader(TensorDataset(train_inputs, train_targets), batch_size=2, shuffle=False),
            "retrain": DataLoader(TensorDataset(retain_inputs, retain_targets), batch_size=2, shuffle=False),
            "retain": DataLoader(TensorDataset(retain_inputs, retain_targets), batch_size=2, shuffle=False),
            "forget": DataLoader(TensorDataset(forget_inputs, forget_targets), batch_size=1, shuffle=False),
            "val": DataLoader(TensorDataset(val_inputs, val_targets), batch_size=2, shuffle=False),
            "test": DataLoader(TensorDataset(test_inputs, test_targets), batch_size=2, shuffle=False),
        }
        return data_utils.DataBundle(
            context=context,
            loaders=loaders,
            class_counts={0: 2, 1: 2},
        )

    def _write_checkpoint_family(
        self,
        checkpoints_root: Path,
        bundle: data_utils.DataBundle,
        *,
        family_name: str,
        num_models: int,
        runtime_seconds: float,
        seed_offset: int = 0,
        model_factory: Any = _tiny_model_factory,
    ) -> Path:
        family_dir = checkpoints_root / bundle.context.dataset / bundle.context.task_id / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(num_models):
            torch.manual_seed(seed + seed_offset)
            model = model_utils.build_model(
                model_factory,
                num_classes=bundle.context.num_classes,
                dataset=bundle.context.dataset,
            )
            checkpoint_path = family_dir / f"seed_{seed}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            metadata = {
                "dataset": bundle.context.dataset,
                "task_id": bundle.context.task_id,
                "run_name": family_name,
                "train_split": "train",
                "seed": seed,
                "runtime_seconds": float(runtime_seconds),
                "checkpoint_path": str(checkpoint_path),
                "num_classes": bundle.context.num_classes,
                "label_to_index": bundle.context.label_to_index,
                "class_names": bundle.context.class_names,
            }
            checkpoint_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return family_dir

    def _write_manifest_files(self, root: Path, *, task_id: str) -> tuple[Path, Path]:
        samples_csv = root / "samples.csv"
        samples_csv.write_text("sample_id,label_id\nplaceholder,0\n", encoding="utf-8")
        task_manifest = root / "task_manifest.json"
        task_manifest.write_text(json.dumps({"task_id": task_id}, indent=2), encoding="utf-8")
        return samples_csv, task_manifest

    def test_run_scrub_unlearning_workflow_creates_checkpoint_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
            )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ):
                outputs = notebook_workflows.run_scrub_unlearning_workflow(
                    dataset="cifar10",
                    base_family_dir=base_family_dir,
                    output_family_name="SCRUB",
                    num_bank_seeds=1,
                    profile="cifar10",
                    checkpoint_dir=checkpoints_root,
                    device_name="cpu",
                    use_wandb=False,
                    image_size=32,
                    reuse_existing=False,
                )
            self.assertEqual(outputs["family_name"], "SCRUB")
            self.assertEqual(len(outputs["seed_bank"]), 1)
            metadata = outputs["seed_bank"][0]
            checkpoint_path = Path(metadata["checkpoint_path"])
            self.assertTrue(checkpoint_path.exists())
            saved_metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["unlearning_algorithm"], "SCRUB")
            self.assertEqual(saved_metadata["algorithm_profile"], "cifar10")
            self.assertTrue(saved_metadata["runtime_excludes_validation"])
            self.assertIn("algorithm_hyperparameters", saved_metadata)
            self.assertEqual(saved_metadata["algorithm_hyperparameters"]["epochs"], 10)
            self.assertTrue(saved_metadata["epochs_logged"])
            self.assertTrue(
                {"forget_loss", "retain_kd_loss", "retain_ce_loss", "val_accuracy", "stage"}.issubset(
                    saved_metadata["epochs_logged"][0]
                )
            )

    def test_run_scrub_unlearning_workflow_reuses_matching_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
            )
            common_kwargs = {
                "dataset": "cifar10",
                "base_family_dir": base_family_dir,
                "output_family_name": "SCRUB",
                "num_bank_seeds": 1,
                "profile": "cifar10",
                "checkpoint_dir": checkpoints_root,
                "device_name": "cpu",
                "use_wandb": False,
                "image_size": 32,
            }
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ):
                notebook_workflows.run_scrub_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=False,
                )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ), mock.patch.object(
                notebook_workflows,
                "load_model_checkpoint",
                side_effect=AssertionError("SCRUB reuse should skip model loading"),
            ):
                reused_outputs = notebook_workflows.run_scrub_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=True,
                )
            self.assertTrue(reused_outputs["seed_bank"][0]["reused_existing"])

    def test_run_delete_unlearning_workflow_creates_checkpoint_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
            )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ):
                outputs = notebook_workflows.run_delete_unlearning_workflow(
                    dataset="cifar10",
                    base_family_dir=base_family_dir,
                    output_family_name="DELETE",
                    num_bank_seeds=1,
                    profile="cifar10",
                    checkpoint_dir=checkpoints_root,
                    device_name="cpu",
                    use_wandb=False,
                    image_size=32,
                    reuse_existing=False,
                )
            self.assertEqual(outputs["family_name"], "DELETE")
            self.assertEqual(len(outputs["seed_bank"]), 1)
            metadata = outputs["seed_bank"][0]
            checkpoint_path = Path(metadata["checkpoint_path"])
            self.assertTrue(checkpoint_path.exists())
            saved_metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["unlearning_algorithm"], "DELETE")
            self.assertEqual(saved_metadata["algorithm_profile"], "cifar10")
            self.assertTrue(saved_metadata["runtime_excludes_validation"])
            self.assertEqual(saved_metadata["algorithm_hyperparameters"]["soft_label"], "inf")
            self.assertTrue(
                {"forget_loss", "val_accuracy", "stage", "learning_rate"}.issubset(
                    saved_metadata["epochs_logged"][0]
                )
            )

    def test_run_delete_unlearning_workflow_reuses_matching_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
            )
            common_kwargs = {
                "dataset": "cifar10",
                "base_family_dir": base_family_dir,
                "output_family_name": "DELETE",
                "num_bank_seeds": 1,
                "profile": "cifar10",
                "checkpoint_dir": checkpoints_root,
                "device_name": "cpu",
                "use_wandb": False,
                "image_size": 32,
            }
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ):
                notebook_workflows.run_delete_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=False,
                )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_model_factory,
            ), mock.patch.object(
                notebook_workflows,
                "load_model_checkpoint",
                side_effect=AssertionError("DELETE reuse should skip model loading"),
            ):
                reused_outputs = notebook_workflows.run_delete_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=True,
                )
            self.assertTrue(reused_outputs["seed_bank"][0]["reused_existing"])

    def test_run_msg_unlearning_workflow_creates_checkpoint_metadata_and_plain_loadable_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
                model_factory=_tiny_conv_model_factory,
            )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ):
                outputs = notebook_workflows.run_msg_unlearning_workflow(
                    dataset="cifar10",
                    base_family_dir=base_family_dir,
                    output_family_name="MSG",
                    num_bank_seeds=1,
                    profile="cifar10",
                    checkpoint_dir=checkpoints_root,
                    device_name="cpu",
                    use_wandb=False,
                    image_size=32,
                    reuse_existing=False,
                )
            self.assertEqual(outputs["family_name"], "MSG")
            self.assertEqual(len(outputs["seed_bank"]), 1)
            metadata = outputs["seed_bank"][0]
            checkpoint_path = Path(metadata["checkpoint_path"])
            self.assertTrue(checkpoint_path.exists())
            saved_metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["unlearning_algorithm"], "MSG")
            self.assertEqual(saved_metadata["algorithm_source_alias"], "KGLTop2")
            self.assertEqual(saved_metadata["algorithm_profile"], "cifar10")
            self.assertTrue(saved_metadata["runtime_excludes_validation"])
            self.assertEqual(saved_metadata["algorithm_hyperparameters"]["init_rate"], 0.3)
            self.assertTrue(
                {"retain_loss", "val_accuracy", "stage", "learning_rate"}.issubset(
                    saved_metadata["epochs_logged"][0]
                )
            )

            plain_model = model_utils.build_model(
                _tiny_conv_model_factory,
                num_classes=bundle.context.num_classes,
                dataset=bundle.context.dataset,
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            plain_model.load_state_dict(state_dict, strict=True)

    def test_run_msg_unlearning_workflow_reuses_matching_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
                model_factory=_tiny_conv_model_factory,
            )
            common_kwargs = {
                "dataset": "cifar10",
                "base_family_dir": base_family_dir,
                "output_family_name": "MSG",
                "num_bank_seeds": 1,
                "profile": "cifar10",
                "checkpoint_dir": checkpoints_root,
                "device_name": "cpu",
                "use_wandb": False,
                "image_size": 32,
            }
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ):
                notebook_workflows.run_msg_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=False,
                )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ), mock.patch.object(
                notebook_workflows,
                "load_model_checkpoint",
                side_effect=AssertionError("MSG reuse should skip model loading"),
            ):
                reused_outputs = notebook_workflows.run_msg_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=True,
                )
            self.assertTrue(reused_outputs["seed_bank"][0]["reused_existing"])

    def test_run_ct_unlearning_workflow_creates_checkpoint_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
                model_factory=_tiny_conv_model_factory,
            )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ):
                outputs = notebook_workflows.run_ct_unlearning_workflow(
                    dataset="cifar10",
                    base_family_dir=base_family_dir,
                    output_family_name="CT",
                    num_bank_seeds=1,
                    profile="cifar10",
                    checkpoint_dir=checkpoints_root,
                    device_name="cpu",
                    use_wandb=False,
                    image_size=32,
                    reuse_existing=False,
                )
            self.assertEqual(outputs["family_name"], "CT")
            self.assertEqual(len(outputs["seed_bank"]), 1)
            metadata = outputs["seed_bank"][0]
            checkpoint_path = Path(metadata["checkpoint_path"])
            self.assertTrue(checkpoint_path.exists())
            saved_metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["unlearning_algorithm"], "CT")
            self.assertEqual(saved_metadata["algorithm_source_alias"], "KGLTop5")
            self.assertEqual(saved_metadata["algorithm_profile"], "cifar10")
            self.assertTrue(saved_metadata["runtime_excludes_validation"])
            self.assertEqual(saved_metadata["algorithm_hyperparameters"]["epochs"], 3)
            self.assertTrue(
                {"retain_loss", "val_accuracy", "stage", "learning_rate"}.issubset(
                    saved_metadata["epochs_logged"][0]
                )
            )

    def test_run_ct_unlearning_workflow_reuses_matching_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            checkpoints_root = root / "checkpoints"
            base_family_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=1,
                runtime_seconds=5.0,
                model_factory=_tiny_conv_model_factory,
            )
            common_kwargs = {
                "dataset": "cifar10",
                "base_family_dir": base_family_dir,
                "output_family_name": "CT",
                "num_bank_seeds": 1,
                "profile": "cifar10",
                "checkpoint_dir": checkpoints_root,
                "device_name": "cpu",
                "use_wandb": False,
                "image_size": 32,
            }
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ):
                notebook_workflows.run_ct_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=False,
                )
            with mock.patch.object(notebook_workflows, "create_dataloaders_from_manifest", return_value=bundle), mock.patch.object(
                notebook_workflows,
                "create_resnet18",
                new=_tiny_conv_model_factory,
            ), mock.patch.object(
                notebook_workflows,
                "load_model_checkpoint",
                side_effect=AssertionError("CT reuse should skip model loading"),
            ):
                reused_outputs = notebook_workflows.run_ct_unlearning_workflow(
                    **common_kwargs,
                    reuse_existing=True,
                )
            self.assertTrue(reused_outputs["seed_bank"][0]["reused_existing"])

    def test_run_benchmark_notebook_workflow_supports_multiple_candidate_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._build_bundle(root)
            samples_csv, task_manifest = self._write_manifest_files(root, task_id=bundle.context.task_id)
            checkpoints_root = root / "checkpoints"
            self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_train",
                num_models=2,
                runtime_seconds=2.0,
                seed_offset=0,
            )
            self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="baseline_retrain",
                num_models=2,
                runtime_seconds=10.0,
                seed_offset=10,
            )
            fanchuan_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="FanchuanUnlearning",
                num_models=2,
                runtime_seconds=1.0,
                seed_offset=20,
            )
            scrub_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="SCRUB",
                num_models=2,
                runtime_seconds=1.5,
                seed_offset=30,
            )
            delete_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="DELETE",
                num_models=2,
                runtime_seconds=1.1,
                seed_offset=40,
            )
            msg_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="MSG",
                num_models=2,
                runtime_seconds=1.2,
                seed_offset=50,
            )
            ct_dir = self._write_checkpoint_family(
                checkpoints_root,
                bundle,
                family_name="CT",
                num_models=2,
                runtime_seconds=0.9,
                seed_offset=60,
            )
            with _pushd(root), mock.patch.object(
                benchmark_utils,
                "create_dataloaders_from_manifest",
                return_value=bundle,
            ), mock.patch.object(
                benchmark_utils,
                "create_resnet18",
                new=_tiny_model_factory,
            ):
                outputs = notebook_workflows.run_benchmark_notebook_workflow(
                    dataset="cifar10",
                    checkpoint_dir=checkpoints_root,
                    baseline_train_family="baseline_train",
                    baseline_retrain_family="baseline_retrain",
                    candidate_family_dirs={
                        "FanchuanUnlearning": fanchuan_dir,
                        "SCRUB": scrub_dir,
                        "DELETE": delete_dir,
                        "MSG": msg_dir,
                        "CT": ct_dir,
                    },
                    efficiency_ratio=0.2,
                    data_root=root,
                    task_manifest=task_manifest,
                    samples_csv=samples_csv,
                    device_name="cpu",
                    batch_size=2,
                    image_size=32,
                    use_wandb=False,
                )
            self.assertIn("FanchuanUnlearning", outputs["family_summaries"])
            self.assertIn("SCRUB", outputs["family_summaries"])
            self.assertIn("DELETE", outputs["family_summaries"])
            self.assertIn("MSG", outputs["family_summaries"])
            self.assertIn("CT", outputs["family_summaries"])
            self.assertIn("FanchuanUnlearning", outputs["comparisons_to_reference"])
            self.assertIn("SCRUB", outputs["comparisons_to_reference"])
            self.assertIn("DELETE", outputs["comparisons_to_reference"])
            self.assertIn("MSG", outputs["comparisons_to_reference"])
            self.assertIn("CT", outputs["comparisons_to_reference"])


if __name__ == "__main__":
    unittest.main()
