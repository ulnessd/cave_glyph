#!/usr/bin/env python3
"""PyQt6 GUI for training and demonstrating a cave glyph CNN.

Version 5 keeps the V4 layout and replaces the replay tab with a genuine held-out test-set workflow.

Version 5 adds:
- live training/validation accuracy plot
- optional Live Demo vs Benchmark mode
- per-epoch sampled validation glyph display
- archaeology-themed display language toggle
- dynamic status notation for classroom use
- final confusion matrix, per-class accuracy, and most-confused pairs
- student test-set evaluation tab
- saved plots and analysis artifacts

Usage:
    python cave_glyph_trainer_gui_v5.py

Worker mode:
    python cave_glyph_trainer_gui_v5.py --worker ...
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_RE = re.compile(r"^G(\d+)(?:_|\b)", re.IGNORECASE)
EVENT_PREFIX = "GLYPH_TRAINER_EVENT "


def emit_event(event_type: str, **payload: object) -> None:
    """Emit a structured event line for the GUI to parse."""
    msg = {"event": event_type, **payload}
    print(EVENT_PREFIX + json.dumps(msg, ensure_ascii=False), flush=True)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str


class WorkerError(RuntimeError):
    pass


# ----------------------------- Worker helpers ----------------------------- #

def discover_images(dataset_dir: Path) -> List[ImageRecord]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise WorkerError(f"Dataset directory does not exist: {dataset_dir}")

    records: List[ImageRecord] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        match = LABEL_RE.match(path.stem)
        if not match:
            continue
        label = f"G{int(match.group(1))}"
        records.append(ImageRecord(path=str(path), label=label))

    if not records:
        raise WorkerError(
            "No glyph image files were found. Expected filenames beginning with G1, G2, ... G10."
        )
    return records


def stratified_split(
    records: Sequence[ImageRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    by_label: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)

    train: List[ImageRecord] = []
    val: List[ImageRecord] = []
    rng = random.Random(seed)

    for label in sorted(by_label.keys(), key=lambda s: int(s[1:])):
        items = list(by_label[label])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        if n_val >= len(items):
            n_val = max(1, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    if not train or not val:
        raise WorkerError("Train/validation split failed. Check dataset size and validation fraction.")
    return train, val


def load_images_to_numpy(records: Sequence[ImageRecord], class_to_index: Dict[str, int], image_size: Tuple[int, int]):
    import numpy as np
    from PIL import Image, ImageOps

    width, height = image_size
    x = np.zeros((len(records), height, width, 1), dtype=np.float32)
    y = np.zeros((len(records),), dtype=np.int64)

    for i, rec in enumerate(records):
        with Image.open(rec.path) as img:
            gray = ImageOps.grayscale(img)
            resized = gray.resize((width, height))
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            x[i, :, :, 0] = arr
            y[i] = class_to_index[rec.label]
    return x, y


def build_model(num_classes: int, image_size: Tuple[int, int]):
    import tensorflow as tf

    width, height = image_size
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def configure_tensorflow(force_cpu: bool) -> str:
    import tensorflow as tf

    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "CPU"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        return f"GPU ({gpus[0].name})"
    return "CPU (no GPU detected by TensorFlow)"


def top_k_labels(prob_row, class_names: Sequence[str], k: int = 3) -> List[Dict[str, object]]:
    import numpy as np

    k = min(k, len(class_names))
    top_idx = np.argsort(prob_row)[::-1][:k]
    return [
        {"label": class_names[int(idx)], "prob": float(prob_row[int(idx)])}
        for idx in top_idx
    ]


def choose_demo_indices(
    probs,
    y_true,
    epoch_number: int,
    rng: random.Random,
) -> List[int]:
    import numpy as np

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    wrong_idx = np.where(~correct)[0].tolist()
    low_conf_correct_idx = np.where(correct)[0].tolist()
    low_conf_correct_idx.sort(key=lambda i: float(conf[i]))

    sample_count = 5 if epoch_number <= 3 else 3
    chosen: List[int] = []

    if wrong_idx:
        wrong_idx.sort(key=lambda i: float(conf[i]), reverse=True)
        chosen.extend(wrong_idx[: min(2, len(wrong_idx), sample_count)])

    for idx in low_conf_correct_idx:
        if len(chosen) >= sample_count:
            break
        if idx not in chosen:
            chosen.append(idx)

    remaining = [i for i in range(len(y_true)) if i not in chosen]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(chosen) >= sample_count:
            break
        chosen.append(idx)

    return chosen[:sample_count]


def make_demo_payload(indices: Sequence[int], probs, y_true, records: Sequence[ImageRecord], class_names: Sequence[str]) -> List[Dict[str, object]]:
    import numpy as np

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    payload: List[Dict[str, object]] = []
    for idx in indices:
        payload.append(
            {
                "path": records[int(idx)].path,
                "true_label": class_names[int(y_true[int(idx)])],
                "pred_label": class_names[int(pred[int(idx)])],
                "confidence": float(conf[int(idx)]),
                "correct": bool(pred[int(idx)] == y_true[int(idx)]),
                "top3": top_k_labels(probs[int(idx)], class_names, k=3),
            }
        )
    return payload


def compute_confusion(y_true, y_pred, num_classes: int):
    import numpy as np

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def per_class_accuracy(confusion, class_names: Sequence[str]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for i, name in enumerate(class_names):
        row_total = int(confusion[i].sum())
        acc = float(confusion[i, i] / row_total) if row_total else 0.0
        results.append({"label": name, "accuracy": acc, "count": row_total})
    return results


def most_confused_pairs(confusion, class_names: Sequence[str], top_n: int = 8) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(confusion[i, j])
            if count > 0:
                pairs.append({"true_label": true_name, "pred_label": pred_name, "count": count})
    pairs.sort(key=lambda d: (-int(d["count"]), d["true_label"], d["pred_label"]))
    return pairs[:top_n]


def hardest_examples(records: Sequence[ImageRecord], probs, y_true, class_names: Sequence[str], limit: int = 40) -> List[Dict[str, object]]:
    import numpy as np

    pred = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    true_conf = probs[np.arange(len(y_true)), y_true]
    examples: List[Dict[str, object]] = []

    for idx in range(len(y_true)):
        correct = bool(pred[idx] == y_true[idx])
        difficulty = 10.0 + float(pred_conf[idx]) if not correct else 1.0 - float(true_conf[idx])
        examples.append(
            {
                "path": records[idx].path,
                "true_label": class_names[int(y_true[idx])],
                "pred_label": class_names[int(pred[idx])],
                "confidence": float(pred_conf[idx]),
                "true_label_confidence": float(true_conf[idx]),
                "correct": correct,
                "difficulty": float(difficulty),
                "top3": top_k_labels(probs[idx], class_names, k=3),
            }
        )

    examples.sort(key=lambda d: (-float(d["difficulty"]), str(d["path"])))
    return examples[:limit]


def save_outputs(
    output_dir: Path,
    class_names: Sequence[str],
    history: Dict[str, List[float]],
    metadata: Dict[str, object],
    final_analysis: Dict[str, object],
    model,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "glyph_cnn_model.keras"
    label_map_path = output_dir / "glyph_label_map.pkl"
    history_path = output_dir / "training_history.json"
    metadata_path = output_dir / "training_metadata.json"
    analysis_path = output_dir / "final_analysis.json"

    model.save(model_path)
    with label_map_path.open("wb") as fh:
        pickle.dump({"class_names": list(class_names)}, fh)
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with analysis_path.open("w", encoding="utf-8") as fh:
        json.dump(final_analysis, fh, indent=2)

    emit_event(
        "artifacts_saved",
        model_path=str(model_path),
        label_map_path=str(label_map_path),
        history_path=str(history_path),
        metadata_path=str(metadata_path),
        analysis_path=str(analysis_path),
    )


def load_saved_model_bundle(model_dir: Path):
    import tensorflow as tf

    model_path = model_dir / "glyph_cnn_model.keras"
    label_map_path = model_dir / "glyph_label_map.pkl"
    metadata_path = model_dir / "training_metadata.json"

    if not model_path.exists():
        raise WorkerError(f"Saved model not found: {model_path}")
    if not label_map_path.exists():
        raise WorkerError(f"Saved label map not found: {label_map_path}")

    model = tf.keras.models.load_model(model_path)
    with label_map_path.open("rb") as fh:
        label_map = pickle.load(fh)
    class_names = list(label_map.get("class_names", []))
    if not class_names:
        raise WorkerError("Saved label map is missing class_names.")

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    width = int(metadata.get("image_size", {}).get("width", 64))
    height = int(metadata.get("image_size", {}).get("height", 64))
    return model, class_names, metadata, (width, height)


def save_test_outputs(model_dir: Path, analysis: Dict[str, object]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = model_dir / "test_set_analysis.json"
    with analysis_path.open("w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)
    emit_event("test_artifacts_saved", analysis_path=str(analysis_path))


def test_worker_main(args: argparse.Namespace) -> int:
    try:
        model_dir = Path(args.model_dir).expanduser().resolve()
        test_dir = Path(args.test_dataset_dir).expanduser().resolve()

        emit_event("log", message=f"Loading saved model bundle from: {model_dir}")
        device_summary = configure_tensorflow(args.force_cpu)
        emit_event("device_selected", device=device_summary)

        model, class_names, metadata, image_size = load_saved_model_bundle(model_dir)
        emit_event("test_model_loaded", labels=class_names, image_size={"width": image_size[0], "height": image_size[1]})
        emit_event("log", message=f"Scanning held-out test set: {test_dir}")
        records = discover_images(test_dir)

        counts: Dict[str, int] = {label: 0 for label in class_names}
        filtered_records: List[ImageRecord] = []
        for rec in records:
            if rec.label in counts:
                counts[rec.label] += 1
                filtered_records.append(rec)
        records = filtered_records
        if not records:
            raise WorkerError("No labeled test-set images matching the trained classes were found.")

        emit_event("test_dataset_found", num_images=len(records), counts=counts)
        class_to_index = {label: i for i, label in enumerate(class_names)}
        x_test, y_test = load_images_to_numpy(records, class_to_index, image_size)
        probs = model.predict(x_test, verbose=0)
        y_pred = probs.argmax(axis=1)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        confusion = compute_confusion(y_test, y_pred, len(class_names))
        class_acc = per_class_accuracy(confusion, class_names)
        confused_pairs = most_confused_pairs(confusion, class_names, top_n=12)

        examples: List[Dict[str, object]] = []
        for idx, rec in enumerate(records):
            pred_label = class_names[int(y_pred[idx])]
            true_label = class_names[int(y_test[idx])]
            conf = float(probs[idx].max())
            examples.append({
                "path": rec.path,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": conf,
                "correct": bool(pred_label == true_label),
                "top3": top_k_labels(probs[idx], class_names, k=3),
            })

        analysis = {
            "labels": class_names,
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "num_images": len(records),
            "counts": counts,
            "device": device_summary,
            "confusion_matrix": confusion.tolist(),
            "per_class_accuracy": class_acc,
            "most_confused_pairs": confused_pairs,
            "examples": examples,
        }

        emit_event("test_complete", **analysis)
        save_test_outputs(model_dir, analysis)
        emit_event("finished_ok")
        return 0
    except Exception as exc:
        emit_event("error", message=str(exc))
        return 1


class EpochEventCallback:
    def __init__(
        self,
        total_epochs: int,
        x_val,
        y_val,
        val_records: Sequence[ImageRecord],
        class_names: Sequence[str],
        demo_mode: bool,
        seed: int,
    ):
        self.total_epochs = total_epochs
        self.x_val = x_val
        self.y_val = y_val
        self.val_records = list(val_records)
        self.class_names = list(class_names)
        self.demo_mode = demo_mode
        self.rng = random.Random(seed)
        self.epoch_start = None
        self.model = None

    def on_train_begin(self, logs=None):
        emit_event("train_begin")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()
        emit_event("epoch_begin", epoch=epoch + 1, total_epochs=self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        duration = None
        if self.epoch_start is not None:
            duration = time.perf_counter() - self.epoch_start
        logs = logs or {}
        emit_event(
            "epoch_end",
            epoch=epoch + 1,
            total_epochs=self.total_epochs,
            epoch_seconds=duration,
            loss=float(logs.get("loss", 0.0)),
            accuracy=float(logs.get("accuracy", 0.0)),
            val_loss=float(logs.get("val_loss", 0.0)),
            val_accuracy=float(logs.get("val_accuracy", 0.0)),
        )
        if self.demo_mode and self.model is not None:
            probs = self.model.predict(self.x_val, verbose=0)
            indices = choose_demo_indices(probs, self.y_val, epoch + 1, self.rng)
            payload = make_demo_payload(indices, probs, self.y_val, self.val_records, self.class_names)
            emit_event("demo_samples", epoch=epoch + 1, samples=payload)

    def on_train_end(self, logs=None):
        emit_event("train_end")


def worker_main(args: argparse.Namespace) -> int:
    try:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        image_size = (args.img_width, args.img_height)

        emit_event("log", message=f"Scanning dataset: {dataset_dir}")
        records = discover_images(dataset_dir)
        labels = sorted({rec.label for rec in records}, key=lambda s: int(s[1:]))
        class_to_index = {label: i for i, label in enumerate(labels)}
        counts: Dict[str, int] = {label: 0 for label in labels}
        for rec in records:
            counts[rec.label] += 1
        emit_event("dataset_found", num_images=len(records), labels=labels, counts=counts)

        train_records, val_records = stratified_split(records, args.val_fraction, args.seed)
        emit_event(
            "split_complete",
            train_count=len(train_records),
            val_count=len(val_records),
            val_fraction=args.val_fraction,
        )

        import tensorflow as tf
        from tensorflow.keras.callbacks import Callback, EarlyStopping

        class TfEventCallback(Callback):
            def __init__(self, inner: EpochEventCallback):
                super().__init__()
                self.inner = inner

            def set_model(self, model):
                super().set_model(model)
                self.inner.model = model

            def on_train_begin(self, logs=None):
                self.inner.on_train_begin(logs)

            def on_epoch_begin(self, epoch, logs=None):
                self.inner.on_epoch_begin(epoch, logs)

            def on_epoch_end(self, epoch, logs=None):
                self.inner.on_epoch_end(epoch, logs)

            def on_train_end(self, logs=None):
                self.inner.on_train_end(logs)

        device_summary = configure_tensorflow(args.force_cpu)
        emit_event("device_selected", device=device_summary)

        emit_event("log", message="Loading images into memory...")
        x_train, y_train = load_images_to_numpy(train_records, class_to_index, image_size)
        x_val, y_val = load_images_to_numpy(val_records, class_to_index, image_size)
        emit_event("numpy_ready", x_train_shape=list(x_train.shape), x_val_shape=list(x_val.shape))

        model = build_model(num_classes=len(labels), image_size=image_size)
        model.summary(print_fn=lambda line: emit_event("log", message=line))

        inner = EpochEventCallback(
            total_epochs=args.epochs,
            x_val=x_val,
            y_val=y_val,
            val_records=val_records,
            class_names=labels,
            demo_mode=bool(args.live_demo_mode),
            seed=args.seed + 99,
        )
        callbacks: List[Callback] = [TfEventCallback(inner)]
        if args.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_accuracy",
                    patience=args.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        start = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        total_seconds = time.perf_counter() - start

        train_probs = model.predict(x_train, verbose=0)
        val_probs = model.predict(x_val, verbose=0)
        train_pred = train_probs.argmax(axis=1)
        val_pred = val_probs.argmax(axis=1)

        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)

        confusion = compute_confusion(y_val, val_pred, len(labels))
        class_acc = per_class_accuracy(confusion, labels)
        confused_pairs = most_confused_pairs(confusion, labels, top_n=8)
        hard_examples = hardest_examples(val_records, val_probs, y_val, labels, limit=40)

        final_analysis = {
            "labels": labels,
            "confusion_matrix": confusion.tolist(),
            "per_class_accuracy": class_acc,
            "most_confused_pairs": confused_pairs,
            "hardest_examples": hard_examples,
        }

        metadata = {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "device": device_summary,
            "image_size": {"width": args.img_width, "height": args.img_height},
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "validation_fraction": args.val_fraction,
            "seed": args.seed,
            "labels": labels,
            "label_counts": counts,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "total_training_seconds": total_seconds,
            "tensorflow_version": tf.__version__,
            "use_early_stopping": bool(args.use_early_stopping),
            "early_stopping_patience": int(args.early_stopping_patience),
            "live_demo_mode": bool(args.live_demo_mode),
        }

        emit_event(
            "training_complete",
            total_seconds=total_seconds,
            train_accuracy=float(train_acc),
            val_accuracy=float(val_acc),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            epochs_completed=len(history.history.get("loss", [])),
            epochs_requested=args.epochs,
            early_stopping_used=bool(args.use_early_stopping),
        )
        emit_event(
            "final_analysis",
            labels=labels,
            confusion_matrix=confusion.tolist(),
            per_class_accuracy=class_acc,
            most_confused_pairs=confused_pairs,
            hardest_examples=hard_examples,
        )

        save_outputs(
            output_dir=output_dir,
            class_names=labels,
            history={k: [float(v) for v in values] for k, values in history.history.items()},
            metadata=metadata,
            final_analysis=final_analysis,
            model=model,
        )
        emit_event("finished_ok")
        return 0

    except Exception as exc:
        emit_event("error", message=str(exc))
        return 1


# ------------------------------- GUI section ------------------------------ #

def gui_main() -> int:
    from PyQt6.QtCore import QProcess, QTimer, Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QSpinBox,
        QDoubleSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Cave Glyph CNN Trainer v5")
            self.resize(1280, 800)
            self.process: QProcess | None = None
            self.elapsed_timer = QTimer(self)
            self.elapsed_timer.timeout.connect(self.update_elapsed)
            self.demo_timer = QTimer(self)
            self.demo_timer.timeout.connect(self.advance_demo_frame)
            self.started_at: float | None = None
            self.current_epoch_count = 0
            self.current_output_dir: Path | None = None

            self.history_epochs: List[int] = []
            self.history_train_acc: List[float] = []
            self.history_val_acc: List[float] = []
            self.history_train_loss: List[float] = []
            self.history_val_loss: List[float] = []
            self.demo_queue: List[Dict[str, object]] = []
            self.demo_index = 0
            self.test_examples: List[Dict[str, object]] = []
            self.last_analysis: Dict[str, object] = {}
            self.active_process_role = "train"

            central = QWidget(self)
            self.setCentralWidget(central)
            root = QHBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            left_panel = QWidget()
            left_panel.setMinimumWidth(380)
            left_panel.setMaximumWidth(460)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(8)
            root.addWidget(left_panel, stretch=0)

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(8)
            root.addWidget(right_panel, stretch=1)

            # Left-side setup area
            setup_group = QGroupBox("Training Setup")
            setup_layout = QGridLayout(setup_group)
            left_layout.addWidget(setup_group)

            self.dataset_edit = QLineEdit()
            self.dataset_edit.setPlaceholderText("Choose folder containing glyph images...")
            dataset_btn = QPushButton("Browse...")
            dataset_btn.clicked.connect(self.choose_dataset_dir)

            self.output_edit = QLineEdit(str(Path.cwd() / "glyph_training_output"))
            output_btn = QPushButton("Browse...")
            output_btn.clicked.connect(self.choose_output_dir)

            self.epochs_spin = QSpinBox()
            self.epochs_spin.setRange(1, 500)
            self.epochs_spin.setValue(15)

            self.batch_spin = QSpinBox()
            self.batch_spin.setRange(1, 2048)
            self.batch_spin.setValue(32)

            self.width_spin = QSpinBox()
            self.width_spin.setRange(16, 1024)
            self.width_spin.setValue(64)

            self.height_spin = QSpinBox()
            self.height_spin.setRange(16, 1024)
            self.height_spin.setValue(64)

            self.val_split_spin = QDoubleSpinBox()
            self.val_split_spin.setRange(0.05, 0.50)
            self.val_split_spin.setDecimals(2)
            self.val_split_spin.setSingleStep(0.01)
            self.val_split_spin.setValue(0.20)

            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 2_000_000_000)
            self.seed_spin.setValue(1234)

            self.force_cpu_checkbox = QCheckBox("Force CPU only (disable NVIDIA GPU for this run)")
            self.force_cpu_checkbox.setChecked(False)

            self.live_demo_checkbox = QCheckBox("Live Demo mode (unchecked = Benchmark mode)")
            self.live_demo_checkbox.setChecked(True)

            self.arch_mode_checkbox = QCheckBox("Archaeology mode")
            self.arch_mode_checkbox.setChecked(True)
            self.arch_mode_checkbox.toggled.connect(self.refresh_demo_text_only)

            self.early_stopping_checkbox = QCheckBox("Use early stopping")
            self.early_stopping_checkbox.setChecked(False)
            self.early_stop_patience_spin = QSpinBox()
            self.early_stop_patience_spin.setRange(1, 50)
            self.early_stop_patience_spin.setValue(3)
            self.early_stop_patience_spin.setEnabled(False)
            self.early_stopping_checkbox.toggled.connect(self.early_stop_patience_spin.setEnabled)

            row = 0
            setup_layout.addWidget(QLabel("Dataset folder"), row, 0)
            setup_layout.addWidget(self.dataset_edit, row, 1)
            setup_layout.addWidget(dataset_btn, row, 2)
            row += 1
            setup_layout.addWidget(QLabel("Output folder"), row, 0)
            setup_layout.addWidget(self.output_edit, row, 1)
            setup_layout.addWidget(output_btn, row, 2)
            row += 1
            setup_layout.addWidget(QLabel("Epochs"), row, 0)
            setup_layout.addWidget(self.epochs_spin, row, 1)
            row += 1
            setup_layout.addWidget(QLabel("Batch size"), row, 0)
            setup_layout.addWidget(self.batch_spin, row, 1)
            row += 1
            setup_layout.addWidget(QLabel("Image width"), row, 0)
            setup_layout.addWidget(self.width_spin, row, 1)
            row += 1
            setup_layout.addWidget(QLabel("Image height"), row, 0)
            setup_layout.addWidget(self.height_spin, row, 1)
            row += 1
            setup_layout.addWidget(QLabel("Validation fraction"), row, 0)
            setup_layout.addWidget(self.val_split_spin, row, 1)
            row += 1
            setup_layout.addWidget(QLabel("Random seed"), row, 0)
            setup_layout.addWidget(self.seed_spin, row, 1)
            row += 1
            setup_layout.addWidget(self.force_cpu_checkbox, row, 0, 1, 3)
            row += 1
            setup_layout.addWidget(self.live_demo_checkbox, row, 0, 1, 3)
            row += 1
            setup_layout.addWidget(self.arch_mode_checkbox, row, 0, 1, 3)
            row += 1
            setup_layout.addWidget(self.early_stopping_checkbox, row, 0)
            setup_layout.addWidget(QLabel("Patience"), row, 1)
            setup_layout.addWidget(self.early_stop_patience_spin, row, 2)

            status_group = QGroupBox("Run Status")
            status_layout = QFormLayout(status_group)
            left_layout.addWidget(status_group)

            self.device_label = QLabel("Not started")
            self.dataset_label = QLabel("No dataset loaded yet")
            self.split_label = QLabel("-")
            self.elapsed_label = QLabel("0.0 s")
            self.metrics_label = QLabel("-")
            self.artifact_label = QLabel("-")
            self.mode_label = QLabel("Live Demo")
            self.status_note_label = QLabel("Awaiting first inscription.")
            self.status_note_label.setWordWrap(True)
            self.status_note_label.setStyleSheet("font-size: 14px; font-weight: 600;")

            status_layout.addRow("Device", self.device_label)
            status_layout.addRow("Dataset", self.dataset_label)
            status_layout.addRow("Train/val split", self.split_label)
            status_layout.addRow("Elapsed", self.elapsed_label)
            status_layout.addRow("Mode", self.mode_label)
            status_layout.addRow("Latest metrics", self.metrics_label)
            status_layout.addRow("Status note", self.status_note_label)
            status_layout.addRow("Artifacts", self.artifact_label)

            progress_box = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            progress_box.addWidget(QLabel("Epoch progress"))
            progress_box.addWidget(self.progress)
            left_layout.addLayout(progress_box)

            btn_row = QHBoxLayout()
            self.start_btn = QPushButton("Start Training")
            self.start_btn.clicked.connect(self.start_training)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_training)
            self.stop_btn.setEnabled(False)
            btn_row.addWidget(self.start_btn)
            btn_row.addWidget(self.stop_btn)
            btn_row.addStretch(1)
            left_layout.addLayout(btn_row)

            self.log_box = QPlainTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setPlaceholderText("Console and worker messages will appear here.")
            left_layout.addWidget(self.log_box, stretch=1)

            tabs = QTabWidget()
            right_layout.addWidget(tabs, stretch=1)

            # Training tab
            training_tab = QWidget()
            tabs.addTab(training_tab, "Training")
            training_layout = QHBoxLayout(training_tab)
            training_layout.setContentsMargins(6, 6, 6, 6)
            training_layout.setSpacing(8)

            left_col = QVBoxLayout()
            training_layout.addLayout(left_col, stretch=5)
            right_col = QVBoxLayout()
            training_layout.addLayout(right_col, stretch=4)

            self.curve_figure = Figure(figsize=(5.2, 3.3))
            self.curve_canvas = FigureCanvas(self.curve_figure)
            left_col.addWidget(self.curve_canvas, stretch=3)

            self.loss_figure = Figure(figsize=(5.2, 2.6))
            self.loss_canvas = FigureCanvas(self.loss_figure)
            left_col.addWidget(self.loss_canvas, stretch=2)

            demo_group = QGroupBox("Epoch Demo")
            demo_layout = QVBoxLayout(demo_group)
            right_col.addWidget(demo_group, stretch=3)

            self.demo_heading = QLabel("Current glyph under review")
            self.demo_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.demo_heading.setStyleSheet("font-size: 16px; font-weight: 700;")
            demo_layout.addWidget(self.demo_heading)

            self.demo_image_label = QLabel("No glyph yet")
            self.demo_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.demo_image_label.setMinimumHeight(180)
            self.demo_image_label.setStyleSheet("border: 1px solid #666; background: #efefef;")
            demo_layout.addWidget(self.demo_image_label, stretch=1)

            self.demo_result_label = QLabel("Waiting for an epoch to finish.")
            self.demo_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.demo_result_label.setWordWrap(True)
            self.demo_result_label.setStyleSheet("font-size: 18px; font-weight: 800;")
            demo_layout.addWidget(self.demo_result_label)

            self.demo_detail_label = QLabel("")
            self.demo_detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.demo_detail_label.setWordWrap(True)
            demo_layout.addWidget(self.demo_detail_label)

            self.demo_top3_box = QPlainTextEdit()
            self.demo_top3_box.setReadOnly(True)
            self.demo_top3_box.setPlaceholderText("Top candidate readings will appear here.")
            demo_layout.addWidget(self.demo_top3_box, stretch=1)


            # Analysis tab
            analysis_tab = QWidget()
            tabs.addTab(analysis_tab, "Analysis")
            analysis_layout = QHBoxLayout(analysis_tab)

            self.conf_figure = Figure(figsize=(5.8, 4.2))
            self.conf_canvas = FigureCanvas(self.conf_figure)
            analysis_layout.addWidget(self.conf_canvas, stretch=3)

            analysis_side = QVBoxLayout()
            analysis_layout.addLayout(analysis_side, stretch=2)

            self.per_class_box = QPlainTextEdit()
            self.per_class_box.setReadOnly(True)
            analysis_side.addWidget(self.per_class_box, stretch=1)

            self.confused_box = QPlainTextEdit()
            self.confused_box.setReadOnly(True)
            analysis_side.addWidget(self.confused_box, stretch=1)

            # Test Set tab
            test_tab = QWidget()
            tabs.addTab(test_tab, "Student Test Set")
            test_layout = QVBoxLayout(test_tab)

            test_controls = QHBoxLayout()
            test_layout.addLayout(test_controls)
            test_controls.addWidget(QLabel("Held-out test folder"))
            self.test_dir_edit = QLineEdit()
            self.test_dir_edit.setPlaceholderText("Choose folder containing student-drawn true test glyphs...")
            test_controls.addWidget(self.test_dir_edit, stretch=1)
            test_browse_btn = QPushButton("Browse...")
            test_browse_btn.clicked.connect(self.choose_test_dir)
            test_controls.addWidget(test_browse_btn)
            self.eval_test_btn = QPushButton("Evaluate Test Set")
            self.eval_test_btn.clicked.connect(self.start_test_evaluation)
            test_controls.addWidget(self.eval_test_btn)

            test_main = QHBoxLayout()
            test_layout.addLayout(test_main, stretch=1)

            test_left = QVBoxLayout()
            test_main.addLayout(test_left, stretch=2)
            test_right = QVBoxLayout()
            test_main.addLayout(test_right, stretch=3)

            self.test_summary_box = QPlainTextEdit()
            self.test_summary_box.setReadOnly(True)
            self.test_summary_box.setPlainText("Run a held-out test set to see image-by-image results, confusion, and summary statistics.")
            test_left.addWidget(self.test_summary_box, stretch=1)

            self.test_example_list = QListWidget()
            self.test_example_list.currentRowChanged.connect(self.show_test_example)
            test_left.addWidget(self.test_example_list, stretch=2)

            self.test_image_label = QLabel("No test image selected")
            self.test_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.test_image_label.setMinimumHeight(220)
            self.test_image_label.setStyleSheet("border: 1px solid #666; background: #efefef;")
            test_right.addWidget(self.test_image_label, stretch=2)

            self.test_detail_box = QPlainTextEdit()
            self.test_detail_box.setReadOnly(True)
            test_right.addWidget(self.test_detail_box, stretch=1)

            self.test_conf_figure = Figure(figsize=(5.0, 3.6))
            self.test_conf_canvas = FigureCanvas(self.test_conf_figure)
            test_right.addWidget(self.test_conf_canvas, stretch=3)

            self.reset_plots()
            self.update_mode_labels()

        # ----------------------- general UI helper methods ----------------------- #
        def append_log(self, text: str) -> None:
            self.log_box.appendPlainText(text)
            self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

        def choose_dataset_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Choose dataset folder")
            if path:
                self.dataset_edit.setText(path)

        def choose_output_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Choose output folder")
            if path:
                self.output_edit.setText(path)

        def choose_test_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Choose held-out test folder")
            if path:
                self.test_dir_edit.setText(path)

        def update_mode_labels(self) -> None:
            mode = "Live Demo" if self.live_demo_checkbox.isChecked() else "Benchmark"
            self.mode_label.setText(mode)
            if self.live_demo_checkbox.isChecked():
                self.demo_heading.setText("Current glyph under review")
            else:
                self.demo_heading.setText("Benchmark mode: live demo paused")
                self.demo_image_label.setText("Live demo disabled for this run.")
                self.demo_result_label.setText("Benchmark mode emphasizes timing and plots.")
                self.demo_detail_label.setText("")
                self.demo_top3_box.setPlainText("")

        def refresh_demo_text_only(self) -> None:
            self.update_mode_labels()
            if self.demo_queue:
                idx = min(self.demo_index, len(self.demo_queue) - 1)
                self.display_demo_sample(self.demo_queue[idx])
            if self.test_examples and self.test_example_list.currentRow() >= 0:
                self.show_test_example(self.test_example_list.currentRow())

        def reset_plots(self) -> None:
            self.history_epochs.clear()
            self.history_train_acc.clear()
            self.history_val_acc.clear()
            self.history_train_loss.clear()
            self.history_val_loss.clear()
            self.draw_curves()
            self.draw_loss_curves()
            self.draw_confusion([], [])
            self.draw_test_confusion([], [])
            self.per_class_box.setPlainText("Per-class accuracy will appear here after training.")
            self.confused_box.setPlainText("Most-confused pairs will appear here after training.")
            self.test_examples = []
            self.test_example_list.clear()
            self.test_detail_box.setPlainText("")
            self.test_image_label.setText("No test image selected")
            self.test_summary_box.setPlainText("Run a held-out test set to see image-by-image results, confusion, and summary statistics.")

        def draw_curves(self) -> None:
            self.curve_figure.clear()
            ax = self.curve_figure.add_subplot(111)
            if self.history_epochs:
                ax.plot(self.history_epochs, self.history_train_acc, label="Train accuracy")
                ax.plot(self.history_epochs, self.history_val_acc, label="Validation accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy by Epoch")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            if self.history_epochs:
                ax.legend()
            self.curve_figure.tight_layout()
            self.curve_canvas.draw_idle()

        def draw_loss_curves(self) -> None:
            self.loss_figure.clear()
            ax = self.loss_figure.add_subplot(111)
            if self.history_epochs:
                ax.plot(self.history_epochs, self.history_train_loss, label="Train loss")
                ax.plot(self.history_epochs, self.history_val_loss, label="Validation loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss by Epoch")
            ax.grid(True, alpha=0.3)
            if self.history_epochs:
                ax.legend()
            self.loss_figure.tight_layout()
            self.loss_canvas.draw_idle()

        def draw_confusion(self, matrix: List[List[int]], labels: Sequence[str]) -> None:
            self.conf_figure.clear()
            ax = self.conf_figure.add_subplot(111)
            if matrix and labels:
                import numpy as np
                arr = np.array(matrix)
                im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
                self.conf_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Validation Confusion Matrix")
                thresh = arr.max() / 2.0 if arr.size else 0
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        ax.text(
                            j,
                            i,
                            str(int(arr[i, j])),
                            ha="center",
                            va="center",
                            color="white" if arr[i, j] > thresh else "black",
                            fontsize=8,
                        )
            else:
                ax.set_title("Validation Confusion Matrix")
                ax.text(0.5, 0.5, "No confusion data yet", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            self.conf_figure.tight_layout()
            self.conf_canvas.draw_idle()

        def draw_test_confusion(self, matrix: List[List[int]], labels: Sequence[str]) -> None:
            self.test_conf_figure.clear()
            ax = self.test_conf_figure.add_subplot(111)
            if matrix and labels:
                import numpy as np
                arr = np.array(matrix)
                im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
                self.test_conf_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Held-out Test Confusion Matrix")
                thresh = arr.max() / 2.0 if arr.size else 0
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", color="white" if arr[i, j] > thresh else "black", fontsize=8)
            else:
                ax.set_title("Held-out Test Confusion Matrix")
                ax.text(0.5, 0.5, "No test-set confusion data yet", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            self.test_conf_figure.tight_layout()
            self.test_conf_canvas.draw_idle()

        def narrative_status(self, epoch: int, total_epochs: int, val_acc: float, val_loss: float) -> str:
            if self.arch_mode_checkbox.isChecked():
                if epoch <= 1 and val_acc < 0.40:
                    return "The reading is still uncertain; the fragments have not yet settled into reliable forms."
                if val_acc < 0.75:
                    return "The model is beginning to distinguish recurring forms in the inscriptional record."
                if val_acc < 0.95:
                    return "The archaeological reading is strengthening; several symbol families are now distinct."
                if val_acc < 0.999:
                    return "The model now recognizes most glyphs reliably, with only a few doubtful readings remaining."
                return "The reading has stabilized into a highly consistent interpretation of the symbol set."
            if epoch <= 1 and val_acc < 0.40:
                return "The model is still uncertain."
            if val_acc < 0.75:
                return "The model is beginning to distinguish recurring forms."
            if val_acc < 0.95:
                return "The model now recognizes many symbols reliably."
            if val_acc < 0.999:
                return "The model now recognizes most symbols reliably."
            return "The model has stabilized around a highly consistent reading."

        # --------------------------- process management -------------------------- #
        def start_training(self) -> None:
            dataset_dir = self.dataset_edit.text().strip()
            output_dir = self.output_edit.text().strip()
            if not dataset_dir:
                QMessageBox.warning(self, "Missing dataset", "Please choose a dataset folder.")
                return
            if self.process is not None:
                QMessageBox.information(self, "Already running", "A training process is already running.")
                return

            self.current_output_dir = Path(output_dir).expanduser()
            self.current_epoch_count = 0
            self.started_at = time.perf_counter()
            self.elapsed_label.setText("0.0 s")
            self.progress.setValue(0)
            self.artifact_label.setText("-")
            self.metrics_label.setText("-")
            self.status_note_label.setText("Preparing the dataset for study.")
            self.log_box.clear()
            self.demo_queue = []
            self.demo_index = 0
            self.reset_plots()
            self.update_mode_labels()

            self.active_process_role = "train"
            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self.read_stdout)
            self.process.readyReadStandardError.connect(self.read_stderr)
            self.process.finished.connect(self.process_finished)

            self.start_btn.setEnabled(False)
            self.eval_test_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.elapsed_timer.start(100)

            args = [
                str(Path(__file__).resolve()),
                "--worker",
                "--dataset-dir",
                dataset_dir,
                "--output-dir",
                output_dir,
                "--epochs",
                str(self.epochs_spin.value()),
                "--batch-size",
                str(self.batch_spin.value()),
                "--img-width",
                str(self.width_spin.value()),
                "--img-height",
                str(self.height_spin.value()),
                "--val-fraction",
                f"{self.val_split_spin.value():.2f}",
                "--seed",
                str(self.seed_spin.value()),
            ]
            if self.force_cpu_checkbox.isChecked():
                args.append("--force-cpu")
            if self.early_stopping_checkbox.isChecked():
                args.extend(["--use-early-stopping", "--early-stopping-patience", str(self.early_stop_patience_spin.value())])
            if self.live_demo_checkbox.isChecked():
                args.append("--live-demo-mode")

            self.append_log("Launching worker process...")
            self.append_log("Command: " + " ".join(args))
            self.process.start(sys.executable, args)
            self.mode_label.setText("Live Demo" if self.live_demo_checkbox.isChecked() else "Benchmark")

        def start_test_evaluation(self) -> None:
            test_dir = self.test_dir_edit.text().strip()
            output_dir = self.output_edit.text().strip()
            if not test_dir:
                QMessageBox.warning(self, "Missing test set", "Please choose a held-out test folder.")
                return
            if self.process is not None:
                QMessageBox.information(self, "Busy", "Please wait for the current worker process to finish.")
                return

            model_dir = Path(output_dir).expanduser()
            model_path = model_dir / "glyph_cnn_model.keras"
            label_map_path = model_dir / "glyph_label_map.pkl"
            if not model_path.exists() or not label_map_path.exists():
                QMessageBox.warning(self, "Missing model", "No saved model bundle was found in the selected output folder. Train first or point the output folder at an existing trained model.")
                return

            self.active_process_role = "test"
            self.test_examples = []
            self.test_example_list.clear()
            self.test_detail_box.setPlainText("")
            self.test_image_label.setText("Evaluating held-out student glyphs...")
            self.test_summary_box.setPlainText("Running held-out test-set evaluation...")
            self.draw_test_confusion([], [])

            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self.read_stdout)
            self.process.readyReadStandardError.connect(self.read_stderr)
            self.process.finished.connect(self.process_finished)

            self.start_btn.setEnabled(False)
            self.eval_test_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            args = [
                str(Path(__file__).resolve()),
                "--worker",
                "--evaluate-model",
                "--model-dir",
                str(model_dir),
                "--test-dataset-dir",
                test_dir,
                "--seed",
                str(self.seed_spin.value()),
            ]
            if self.force_cpu_checkbox.isChecked():
                args.append("--force-cpu")

            self.append_log("Launching held-out test evaluation worker...")
            self.append_log("Command: " + " ".join(args))
            self.process.start(sys.executable, args)

        def stop_training(self) -> None:
            if self.process is not None:
                self.append_log("Stopping worker process...")
                self.process.kill()

        def process_finished(self, exit_code: int, exit_status) -> None:
            self.elapsed_timer.stop()
            self.demo_timer.stop()
            if self.process is not None:
                self.append_log(f"Worker exited with code {exit_code}.")
                self.process.deleteLater()
                self.process = None
            self.start_btn.setEnabled(True)
            self.eval_test_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress.setValue(100 if exit_code == 0 else self.progress.value())
            self.save_visual_artifacts()

        def update_elapsed(self) -> None:
            if self.started_at is not None:
                elapsed = time.perf_counter() - self.started_at
                self.elapsed_label.setText(f"{elapsed:.1f} s")

        # ------------------------------- IO parsing ------------------------------ #
        def read_stdout(self) -> None:
            if self.process is None:
                return
            raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            for line in raw.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith(EVENT_PREFIX):
                    payload = json.loads(line[len(EVENT_PREFIX):])
                    self.handle_event(payload)
                else:
                    self.append_log(line)

        def read_stderr(self) -> None:
            if self.process is None:
                return
            raw = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            for line in raw.splitlines():
                self.append_log("[stderr] " + line)

        def handle_event(self, payload: Dict[str, object]) -> None:
            event = str(payload.get("event", ""))
            if event == "log":
                self.append_log(str(payload.get("message", "")))
            elif event == "dataset_found":
                num_images = int(payload.get("num_images", 0))
                counts = payload.get("counts", {})
                labels = payload.get("labels", [])
                summary = f"{num_images} images across {len(labels)} classes"
                self.dataset_label.setText(summary)
                for label in labels:
                    self.append_log(f"  {label}: {counts.get(label, 0)}")
            elif event == "split_complete":
                self.split_label.setText(
                    f"train={payload.get('train_count')}  val={payload.get('val_count')}  split={payload.get('val_fraction')}"
                )
            elif event == "device_selected":
                self.device_label.setText(str(payload.get("device", "?")))
            elif event == "epoch_begin":
                epoch = int(payload.get("epoch", 0))
                total = int(payload.get("total_epochs", 1))
                self.current_epoch_count = epoch
                self.status_note_label.setText(f"Epoch {epoch}/{total}: examining another layer of recurring forms.")
            elif event == "epoch_end":
                epoch = int(payload.get("epoch", 0))
                total = int(payload.get("total_epochs", 1))
                train_acc = float(payload.get("accuracy", 0.0))
                val_acc = float(payload.get("val_accuracy", 0.0))
                train_loss = float(payload.get("loss", 0.0))
                val_loss = float(payload.get("val_loss", 0.0))
                self.history_epochs.append(epoch)
                self.history_train_acc.append(train_acc)
                self.history_val_acc.append(val_acc)
                self.history_train_loss.append(train_loss)
                self.history_val_loss.append(val_loss)
                self.draw_curves()
                self.draw_loss_curves()
                pct = int(round(epoch / max(total, 1) * 100))
                self.progress.setValue(max(0, min(100, pct)))
                self.metrics_label.setText(
                    f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                self.status_note_label.setText(self.narrative_status(epoch, total, val_acc, val_loss))
            elif event == "demo_samples":
                if self.live_demo_checkbox.isChecked():
                    self.demo_queue = list(payload.get("samples", []))
                    self.demo_index = -1
                    self.advance_demo_frame()
                    if self.demo_queue:
                        self.demo_timer.start(1100)
            elif event == "training_complete":
                total_seconds = float(payload.get("total_seconds", 0.0))
                train_acc = float(payload.get("train_accuracy", 0.0))
                val_acc = float(payload.get("val_accuracy", 0.0))
                epochs_completed = int(payload.get("epochs_completed", 0))
                epochs_requested = int(payload.get("epochs_requested", 0))
                self.metrics_label.setText(
                    f"final_train_acc={train_acc:.4f}, final_val_acc={val_acc:.4f}, epochs={epochs_completed}/{epochs_requested}"
                )
                self.elapsed_label.setText(f"{total_seconds:.1f} s")
                self.progress.setValue(100)
                self.status_note_label.setText(self.narrative_status(epochs_completed, max(epochs_requested, 1), val_acc, float(payload.get("val_loss", 0.0))))
                self.append_log(f"Training complete in {total_seconds:.2f} s")
            elif event == "final_analysis":
                self.last_analysis = payload
                labels = payload.get("labels", [])
                matrix = payload.get("confusion_matrix", [])
                self.draw_confusion(matrix, labels)
                per_class = payload.get("per_class_accuracy", [])
                confused_pairs = payload.get("most_confused_pairs", [])
                self.populate_analysis_text(per_class, confused_pairs)
            elif event == "test_dataset_found":
                self.append_log(f"Held-out test set: {payload.get('num_images', 0)} images")
            elif event == "test_model_loaded":
                image_size = payload.get("image_size", {})
                self.append_log(f"Loaded saved model for test evaluation. Image size: {image_size.get('width', '?')}x{image_size.get('height', '?')}")
            elif event == "test_complete":
                self.handle_test_complete(payload)
            elif event == "artifacts_saved":
                model_path = str(payload.get("model_path", "-"))
                self.artifact_label.setText(model_path)
                self.append_log("Artifacts saved:")
                for key in ["model_path", "label_map_path", "history_path", "metadata_path", "analysis_path"]:
                    if key in payload:
                        self.append_log(f"  {key}: {payload[key]}")
            elif event == "test_artifacts_saved":
                self.append_log(f"Held-out test analysis saved: {payload.get('analysis_path', '-')}")
            elif event == "finished_ok":
                self.append_log("Worker finished successfully.")
            elif event == "error":
                msg = str(payload.get("message", "Unknown worker error"))
                self.append_log("ERROR: " + msg)
                QMessageBox.critical(self, "Worker error", msg)
            else:
                self.append_log(f"[unhandled event] {payload}")

        # ---------------------------- demo / test-set UI --------------------------- #
        def display_demo_sample(self, sample: Dict[str, object]) -> None:
            path = str(sample.get("path", ""))
            self.set_image_on_label(self.demo_image_label, path, max_width=320, max_height=260)
            correct = bool(sample.get("correct", False))
            true_label = str(sample.get("true_label", "?"))
            pred_label = str(sample.get("pred_label", "?"))
            conf = float(sample.get("confidence", 0.0))

            if self.arch_mode_checkbox.isChecked():
                heading = "Correct" if correct else "Wrong"
                self.demo_result_label.setText(f"Assessment: {heading}")
                self.demo_detail_label.setText(
                    f"Model interpretation: {pred_label} | Reference reading: {true_label} | Confidence: {conf:.3f}"
                )
            else:
                heading = "Correct" if correct else "Wrong"
                self.demo_result_label.setText(heading)
                self.demo_detail_label.setText(
                    f"Predicted: {pred_label} | True: {true_label} | Confidence: {conf:.3f}"
                )
            color = "#1f7a1f" if correct else "#b22222"
            self.demo_result_label.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {color};")

            top3 = sample.get("top3", [])
            lines = []
            if self.arch_mode_checkbox.isChecked():
                lines.append("Top candidate readings")
            else:
                lines.append("Top predictions")
            for item in top3:
                lines.append(f"{item['label']}: {float(item['prob']):.3f}")
            self.demo_top3_box.setPlainText("\n".join(lines))

        def advance_demo_frame(self) -> None:
            if not self.demo_queue:
                self.demo_timer.stop()
                return
            self.demo_index += 1
            if self.demo_index >= len(self.demo_queue):
                self.demo_timer.stop()
                return
            self.display_demo_sample(self.demo_queue[self.demo_index])

        def populate_analysis_text(self, per_class: Sequence[Dict[str, object]], confused_pairs: Sequence[Dict[str, object]]) -> None:
            class_lines = ["Per-class validation accuracy"]
            for item in per_class:
                class_lines.append(f"{item['label']}: {float(item['accuracy']):.3f}  (n={int(item['count'])})")
            self.per_class_box.setPlainText("\n".join(class_lines))

            pair_lines = ["Most-confused glyph pairs"]
            if confused_pairs:
                for item in confused_pairs:
                    pair_lines.append(f"True {item['true_label']} → predicted {item['pred_label']}: {int(item['count'])}")
            else:
                pair_lines.append("No off-diagonal confusion was observed on the validation set.")
            self.confused_box.setPlainText("\n".join(pair_lines))

        def handle_test_complete(self, payload: Dict[str, object]) -> None:
            labels = list(payload.get("labels", []))
            matrix = list(payload.get("confusion_matrix", []))
            self.draw_test_confusion(matrix, labels)
            self.test_examples = list(payload.get("examples", []))
            self.populate_test_examples()

            acc = float(payload.get("test_accuracy", 0.0))
            loss = float(payload.get("test_loss", 0.0))
            num_images = int(payload.get("num_images", 0))
            counts = payload.get("counts", {})
            confused_pairs = list(payload.get("most_confused_pairs", []))
            per_class = list(payload.get("per_class_accuracy", []))
            num_correct = sum(1 for ex in self.test_examples if bool(ex.get("correct", False)))
            num_wrong = num_images - num_correct

            lines = [
                "Held-out test-set summary",
                f"Images: {num_images}",
                f"Correct: {num_correct}",
                f"Wrong: {num_wrong}",
                f"Accuracy: {acc:.4f}",
                f"Loss: {loss:.4f}",
                "",
                "Counts by class",
            ]
            for label in labels:
                lines.append(f"{label}: {int(counts.get(label, 0))}")
            lines.append("")
            lines.append("Per-class test accuracy")
            for item in per_class:
                lines.append(f"{item['label']}: {float(item['accuracy']):.3f}  (n={int(item['count'])})")
            lines.append("")
            lines.append("Most-confused pairs")
            if confused_pairs:
                for item in confused_pairs:
                    lines.append(f"True {item['true_label']} → predicted {item['pred_label']}: {int(item['count'])}")
            else:
                lines.append("No wrong cases were observed on this held-out test set.")
            self.test_summary_box.setPlainText("\n".join(lines))

        def populate_test_examples(self) -> None:
            self.test_example_list.clear()
            for idx, ex in enumerate(self.test_examples, start=1):
                marker = "✓" if bool(ex.get("correct", False)) else "✗"
                self.test_example_list.addItem(QListWidgetItem(f"{idx:02d}. {marker} {ex.get('true_label')} → {ex.get('pred_label')}  conf={float(ex.get('confidence', 0.0)):.3f}"))
            if self.test_examples:
                self.test_example_list.setCurrentRow(0)
            else:
                self.test_image_label.setText("No evaluated examples yet")
                self.test_detail_box.setPlainText("")

        def show_test_example(self, row: int) -> None:
            if row < 0 or row >= len(self.test_examples):
                return
            ex = self.test_examples[row]
            self.set_image_on_label(self.test_image_label, str(ex.get("path", "")), max_width=420, max_height=300)
            correct = bool(ex.get("correct", False))
            lines = []
            if self.arch_mode_checkbox.isChecked():
                lines.append(f"Assessment: {'Correct' if correct else 'Wrong'}")
                lines.append(f"Reference reading: {ex.get('true_label')}")
                lines.append(f"Model interpretation: {ex.get('pred_label')}")
            else:
                lines.append(f"Correct: {correct}")
                lines.append(f"True label: {ex.get('true_label')}")
                lines.append(f"Predicted label: {ex.get('pred_label')}")
            lines.append(f"Confidence: {float(ex.get('confidence', 0.0)):.4f}")
            lines.append("")
            lines.append("Top candidate readings" if self.arch_mode_checkbox.isChecked() else "Top predictions")
            for item in ex.get("top3", []):
                lines.append(f"{item['label']}: {float(item['prob']):.4f}")
            self.test_detail_box.setPlainText("\n".join(lines))

        def set_image_on_label(self, label: QLabel, path: str, max_width: int, max_height: int) -> None:
            if not path or not Path(path).exists():
                label.setText("Image unavailable")
                label.setPixmap(QPixmap())
                return
            pixmap = QPixmap(path)
            if pixmap.isNull():
                label.setText("Could not load image")
                label.setPixmap(QPixmap())
                return
            scaled = pixmap.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
            label.setText("")

        # ------------------------------- artifact save --------------------------- #
        def save_visual_artifacts(self) -> None:
            if self.current_output_dir is None:
                return
            try:
                self.current_output_dir.mkdir(parents=True, exist_ok=True)
                curve_path = self.current_output_dir / "training_curves.png"
                loss_path = self.current_output_dir / "loss_curves.png"
                conf_path = self.current_output_dir / "confusion_matrix.png"
                test_conf_path = self.current_output_dir / "test_confusion_matrix.png"
                self.curve_figure.savefig(curve_path, dpi=150, bbox_inches="tight")
                self.loss_figure.savefig(loss_path, dpi=150, bbox_inches="tight")
                self.conf_figure.savefig(conf_path, dpi=150, bbox_inches="tight")
                self.test_conf_figure.savefig(test_conf_path, dpi=150, bbox_inches="tight")
                self.append_log(f"Saved plot: {curve_path}")
                self.append_log(f"Saved plot: {loss_path}")
                self.append_log(f"Saved plot: {conf_path}")
                self.append_log(f"Saved plot: {test_conf_path}")
            except Exception as exc:
                self.append_log(f"Could not save plot artifacts: {exc}")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


# ------------------------------- CLI parsing ------------------------------ #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cave glyph CNN trainer GUI / worker / test evaluator")
    parser.add_argument("--worker", action="store_true", help="Run in background worker mode")
    parser.add_argument("--dataset-dir", default="", help="Directory containing glyph images")
    parser.add_argument("--output-dir", default="glyph_training_output", help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=64)
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--use-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--live-demo-mode", action="store_true")
    parser.add_argument("--evaluate-model", action="store_true", help="Evaluate a saved model on a held-out test set")
    parser.add_argument("--model-dir", default="glyph_training_output", help="Directory containing glyph_cnn_model.keras and label map")
    parser.add_argument("--test-dataset-dir", default="", help="Directory containing held-out labeled test images")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.worker and args.evaluate_model:
        return test_worker_main(args)
    if args.worker:
        return worker_main(args)
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
