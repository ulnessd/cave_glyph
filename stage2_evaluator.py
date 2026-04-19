#!/usr/bin/env python3
"""Stage 2 Explorer — Student View v3.

This version is intentionally *structural* rather than semantic. It does not
use cave-art captions or infer meanings. Students explore recurring glyph
chunks using only inscription streams.

Live controls:
- PMI threshold
- Position rigidity penalty
- Multi-glyph penalty

Compared with the earlier student prototype, this version adds a more selective
backend without using the hidden vocabulary:
- stronger anti-3+-glyph bias
- a closure/substring test for longer chunks
- a continuation/branching penalty
- a modest repeated-sign bonus for chunks like ||

The goal is not to solve the lexicon perfectly, but to produce a cleaner pool of
structural candidates for later Stage 3 semantic grounding.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


GLYPH_DISPLAY = {
    "G1": "|",
    "G2": "○",
    "G3": "···",
    "G4": "∩",
    "G5": "†",
    "G6": "∼",
    "G7": "∧",
    "G8": "⅁",
    "G9": "Ⅎ",
    "G10": "⋌",
}


@dataclass
class Candidate:
    chunk_ids: Tuple[str, ...]
    chunk_display: str
    length: int
    count: int
    avg_pmi: float
    min_pmi: float
    start_count: int
    mid_count: int
    end_count: int
    rigidity: float
    left_diversity: int
    right_diversity: int
    left_entropy: float
    right_entropy: float
    branching: float
    closure: float
    extension_pressure: float
    repeated_bonus: float
    score: float
    accepted: bool
    kind: str
    examples: List[int] = field(default_factory=list)


class DatasetModel:
    def __init__(self) -> None:
        self.dataset_path: str = ""
        self.dataset_name: str = ""
        self.sentences: List[dict] = []
        self.stream_records: List[dict] = []
        self.streams: List[List[str]] = []

        self.single_counts: Counter[str] = Counter()
        self.pair_counts: Counter[Tuple[str, str]] = Counter()
        self.chunk_counts: Counter[Tuple[str, ...]] = Counter()
        self.left_ctx: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.right_ctx: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.pos_counts: Dict[Tuple[str, ...], List[int]] = defaultdict(lambda: [0, 0, 0])
        self.examples: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        self.max_chunk_len = 5
        self.total_tokens = 0
        self.total_pairs = 0

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dataset_path = path
        self.dataset_name = data.get("dataset_name", os.path.basename(path))
        self.sentences = data["sentences"]

        self.stream_records = []
        self.streams = []
        self.single_counts.clear()
        self.pair_counts.clear()
        self.chunk_counts.clear()
        self.left_ctx.clear()
        self.right_ctx.clear()
        self.pos_counts.clear()
        self.examples.clear()

        for sent in self.sentences:
            sv = sent["student_view"]
            stream = list(sv["atomic_glyph_stream_ids"])
            self.streams.append(stream)
            rec = {
                "sentence_number": sent["sentence_number"],
                "stream": stream,
                "glyph_sentence": self.display_stream(stream),
                "panel_id": sv.get("linked_panel_id") or "(loose)",
            }
            self.stream_records.append(rec)

            for g in stream:
                self.single_counts[g] += 1
            for a, b in zip(stream, stream[1:]):
                self.pair_counts[(a, b)] += 1

            n = len(stream)
            for i in range(n):
                for L in range(1, self.max_chunk_len + 1):
                    if i + L > n:
                        continue
                    chunk = tuple(stream[i:i+L])
                    self.chunk_counts[chunk] += 1
                    if len(self.examples[chunk]) < 12:
                        self.examples[chunk].append(sent["sentence_number"])

                    left = stream[i-1] if i > 0 else "<START>"
                    right = stream[i+L] if i + L < n else "<END>"
                    self.left_ctx[chunk][left] += 1
                    self.right_ctx[chunk][right] += 1

                    if i == 0:
                        self.pos_counts[chunk][0] += 1
                    elif i + L == n:
                        self.pos_counts[chunk][2] += 1
                    else:
                        self.pos_counts[chunk][1] += 1

        self.total_tokens = sum(self.single_counts.values())
        self.total_pairs = sum(self.pair_counts.values())

    def display_stream(self, glyph_ids: List[str] | Tuple[str, ...]) -> str:
        return " ".join(GLYPH_DISPLAY.get(g, g) for g in glyph_ids)

    @staticmethod
    def entropy(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total <= 0:
            return 0.0
        out = 0.0
        for c in counter.values():
            p = c / total
            out -= p * math.log(p, 2)
        return out

    def pair_pmi(self, a: str, b: str) -> float:
        pair_c = self.pair_counts.get((a, b), 0)
        if pair_c <= 0 or self.total_pairs <= 0:
            return -10.0
        p_ab = pair_c / self.total_pairs
        p_a = self.single_counts[a] / self.total_tokens
        p_b = self.single_counts[b] / self.total_tokens
        denom = p_a * p_b
        if denom <= 0:
            return -10.0
        return math.log(p_ab / denom, 2)

    def following_total(self, glyph_id: str) -> int:
        return sum(c for (a, _), c in self.pair_counts.items() if a == glyph_id)

    def preceding_total(self, glyph_id: str) -> int:
        return sum(c for (_, b), c in self.pair_counts.items() if b == glyph_id)

    def extension_pressure(self, chunk: Tuple[str, ...]) -> float:
        count = self.chunk_counts[chunk]
        if count <= 0:
            return 0.0
        longer = 0
        lefts = {x for x in self.left_ctx[chunk] if x != "<START>"}
        rights = {x for x in self.right_ctx[chunk] if x != "<END>"}
        longer += max(0, len(lefts) - 1)
        longer += max(0, len(rights) - 1)
        return min(2.0, longer / 4.0)

    def closure_score(self, chunk: Tuple[str, ...]) -> float:
        """How self-contained does the chunk look? Higher is better.

        For bigrams, closure combines forward and backward dominance. For longer
        chunks, closure compares the chunk's frequency to immediate prefix and
        suffix frequencies, which is an honest way to ask whether the longer
        chunk behaves like a stable unit rather than merely a common extension.
        """
        count = self.chunk_counts[chunk]
        if count <= 0:
            return 0.0

        if len(chunk) == 1:
            return 1.0

        if len(chunk) == 2:
            a, b = chunk
            fwd_total = max(1, self.following_total(a))
            rev_total = max(1, self.preceding_total(b))
            fwd = count / fwd_total
            rev = count / rev_total
            return max(0.0, min(1.0, 0.5 * (fwd + rev)))

        prefix = chunk[:-1]
        suffix = chunk[1:]
        prefix_count = self.chunk_counts.get(prefix, count)
        suffix_count = self.chunk_counts.get(suffix, count)
        ratio_p = count / max(1, prefix_count)
        ratio_s = count / max(1, suffix_count)
        return max(0.0, min(1.0, 0.5 * (ratio_p + ratio_s)))

    def repeated_bonus(self, chunk: Tuple[str, ...]) -> float:
        # Repeated-sign compounds such as || can be intentional cultural marks.
        # This bonus is only modest; it helps them compete without forcing them.
        if len(chunk) == 2 and chunk[0] == chunk[1]:
            return 0.55
        return 0.0

    def chunk_metrics(self, chunk: Tuple[str, ...], pmi_threshold: float,
                      rigidity_penalty: float, multi_penalty: float,
                      include_repeated_pairs: bool = False) -> Candidate:
        count = self.chunk_counts[chunk]
        length = len(chunk)

        pmis = [self.pair_pmi(a, b) for a, b in zip(chunk, chunk[1:])] if length >= 2 else []
        avg_pmi = sum(pmis) / len(pmis) if pmis else 0.0
        min_pmi = min(pmis) if pmis else 0.0

        start_c, mid_c, end_c = self.pos_counts[chunk]
        rigidity = max(start_c, mid_c, end_c) / count if count else 0.0
        left_div = len(self.left_ctx[chunk])
        right_div = len(self.right_ctx[chunk])
        left_ent = self.entropy(self.left_ctx[chunk])
        right_ent = self.entropy(self.right_ctx[chunk])
        branching = max(0.0, ((left_div - 1) + (right_div - 1) - 2) / 6.0)
        closure = self.closure_score(chunk)
        ext_pressure = self.extension_pressure(chunk)
        repeated = self.repeated_bonus(chunk)

        support = math.log2(count + 1)
        internal_strength = max(0.0, avg_pmi) + 0.35 * max(0.0, min_pmi)

        score = support * internal_strength
        score += 2.2 * closure
        score += 1.6 * repeated
        score -= rigidity_penalty * max(0.0, rigidity - 0.55) * (1.0 + branching)
        score -= 0.8 * rigidity_penalty * ext_pressure
        score -= multi_penalty * max(0, length - 2) ** 2
        score -= 1.15 * multi_penalty * max(0, length - 2) * max(0.0, 0.78 - closure)
        score -= 0.8 * multi_penalty * max(0.0, branching - 0.6)

        # Acceptance rules are intentionally conservative for 3+ chunks.
        if length == 1:
            accepted = True
            kind = "atomic glyph"
        elif length == 2:
            effective_pmi = avg_pmi + repeated
            repeated_pair = chunk[0] == chunk[1]
            forced_repeated = include_repeated_pairs and repeated_pair and count >= 3
            accepted = forced_repeated or ((effective_pmi >= pmi_threshold) and (score >= 3.0) and (closure >= 0.22))
            if accepted:
                kind = "repeated-sign candidate" if forced_repeated else "likely lexical pair"
            else:
                kind = "rejected pair"
        else:
            accepted = (
                avg_pmi >= pmi_threshold + 0.18
                and min_pmi >= max(-0.05, pmi_threshold - 0.35)
                and score >= 4.2
                and closure >= 0.46
                and rigidity <= 0.92
            )
            if accepted:
                kind = "likely formulaic chunk"
            else:
                kind = "rejected longer chunk"

        return Candidate(
            chunk_ids=chunk,
            chunk_display=self.display_stream(chunk),
            length=length,
            count=count,
            avg_pmi=avg_pmi,
            min_pmi=min_pmi,
            start_count=start_c,
            mid_count=mid_c,
            end_count=end_c,
            rigidity=rigidity,
            left_diversity=left_div,
            right_diversity=right_div,
            left_entropy=left_ent,
            right_entropy=right_ent,
            branching=branching,
            closure=closure,
            extension_pressure=ext_pressure,
            repeated_bonus=repeated,
            score=score,
            accepted=accepted,
            kind=kind,
            examples=self.examples[chunk][:],
        )

    def candidate_inventory(self, pmi_threshold: float, rigidity_penalty: float,
                            multi_penalty: float, include_repeated_pairs: bool = False) -> List[Candidate]:
        out: List[Candidate] = []
        for chunk in self.chunk_counts:
            cand = self.chunk_metrics(chunk, pmi_threshold, rigidity_penalty, multi_penalty, include_repeated_pairs)
            if cand.accepted:
                out.append(cand)
        out.sort(key=lambda c: (-c.score, -c.avg_pmi, -c.count, c.chunk_display))
        return out

    def example_rows(self, chunk: Tuple[str, ...]) -> List[Tuple[int, str, str]]:
        sids = set(self.examples.get(chunk, []))
        rows: List[Tuple[int, str, str]] = []
        for rec in self.stream_records:
            if rec["sentence_number"] in sids:
                rows.append((rec["sentence_number"], rec["glyph_sentence"], rec["panel_id"]))
        return rows[:12]


class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 5, height: float = 4, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)


class Stage2StudentWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 2 Explorer — Student View v3")
        self.resize(1360, 860)

        self.model = DatasetModel()
        self.candidates: List[Candidate] = []

        self.dataset_edit = QLineEdit()
        default_dataset = "/mnt/data/cave_script_master_dataset_sentence_centered_restructured.json"
        if os.path.exists(default_dataset):
            self.dataset_edit.setText(default_dataset)

        self.pmi_slider, self.pmi_value = self.make_slider(0, 250, 60, "0.60")
        self.rig_slider, self.rig_value = self.make_slider(0, 300, 120, "1.20")
        self.multi_slider, self.multi_value = self.make_slider(0, 300, 95, "0.95")
        self.repeats_checkbox = QCheckBox("Include repeated-glyph compounds")
        self.repeats_checkbox.setChecked(False)

        self.bar_canvas = MplCanvas(width=7, height=4)
        self.len_canvas = MplCanvas(width=4, height=3)
        self.pos_canvas = MplCanvas(width=4, height=3)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Chunk", "Len", "Count", "Avg PMI", "Min PMI", "Closure",
            "Rigidity", "Score", "Type"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.update_detail_panel)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.examples_table = QTableWidget(0, 3)
        self.examples_table.setHorizontalHeaderLabels(["Sentence #", "Glyph sentence", "Panel"])
        self.examples_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.examples_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(120)

        self.init_ui()
        self.connect_events()

        if self.dataset_edit.text() and os.path.exists(self.dataset_edit.text()):
            self.load_dataset()

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        left = QFrame()
        left.setFrameShape(QFrame.Shape.StyledPanel)
        left.setMaximumWidth(360)
        left_layout = QVBoxLayout(left)

        load_group = QGroupBox("Dataset")
        lg = QGridLayout(load_group)
        lg.addWidget(QLabel("Master JSON"), 0, 0)
        lg.addWidget(self.dataset_edit, 0, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_dataset)
        lg.addWidget(browse_btn, 0, 2)
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self.load_dataset)
        lg.addWidget(load_btn, 1, 1, 1, 2)
        left_layout.addWidget(load_group)

        sliders_group = QGroupBox("Live filters")
        sg = QGridLayout(sliders_group)
        sg.addWidget(QLabel("PMI threshold"), 0, 0)
        sg.addWidget(self.pmi_slider, 0, 1)
        sg.addWidget(self.pmi_value, 0, 2)
        sg.addWidget(QLabel("Position rigidity penalty"), 1, 0)
        sg.addWidget(self.rig_slider, 1, 1)
        sg.addWidget(self.rig_value, 1, 2)
        sg.addWidget(QLabel("Multi-glyph penalty"), 2, 0)
        sg.addWidget(self.multi_slider, 2, 1)
        sg.addWidget(self.multi_value, 2, 2)
        sg.addWidget(self.repeats_checkbox, 3, 0, 1, 3)
        note = QLabel(
            "This version is purely structural. It does not use cave-art captions or\n"
            "suggest meanings. Longer chunks face a stronger closure test. Repeated\n"
            "signs like || can optionally be promoted as intentional candidates."
        )
        note.setWordWrap(True)
        sg.addWidget(note, 4, 0, 1, 3)
        left_layout.addWidget(sliders_group)

        action_group = QGroupBox("Actions")
        ag = QVBoxLayout(action_group)
        refresh_btn = QPushButton("Refresh candidate pool")
        refresh_btn.clicked.connect(self.recompute)
        export_json_btn = QPushButton("Export accepted candidates to JSON")
        export_json_btn.clicked.connect(self.export_json)
        export_csv_btn = QPushButton("Export accepted candidates to CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        ag.addWidget(refresh_btn)
        ag.addWidget(export_json_btn)
        ag.addWidget(export_csv_btn)
        left_layout.addWidget(action_group)

        left_layout.addWidget(QLabel("Session notes"))
        left_layout.addWidget(self.status_box, stretch=1)
        outer.addWidget(left)

        right_split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(right_split, stretch=1)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.bar_canvas, stretch=3)

        pies = QWidget()
        pies_layout = QVBoxLayout(pies)
        pies_layout.setContentsMargins(0, 0, 0, 0)
        pies_layout.addWidget(self.len_canvas)
        pies_layout.addWidget(self.pos_canvas)
        top_layout.addWidget(pies, stretch=2)
        right_split.addWidget(top_widget)

        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.addWidget(bottom_split)

        table_box = QWidget()
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QLabel("Accepted candidate pool"))
        table_layout.addWidget(self.table)
        bottom_split.addWidget(table_box)

        detail_box = QWidget()
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(QLabel("Selected chunk details"))
        detail_layout.addWidget(self.detail_text, stretch=1)
        detail_layout.addWidget(QLabel("Example inscriptions"))
        detail_layout.addWidget(self.examples_table, stretch=1)
        bottom_split.addWidget(detail_box)

        right_split.setSizes([340, 500])
        bottom_split.setSizes([720, 520])

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open dataset…", self)
        open_action.triggered.connect(self.browse_dataset)
        file_menu.addAction(open_action)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def connect_events(self) -> None:
        self.pmi_slider.valueChanged.connect(self.recompute)
        self.rig_slider.valueChanged.connect(self.recompute)
        self.multi_slider.valueChanged.connect(self.recompute)
        self.repeats_checkbox.stateChanged.connect(self.recompute)

    def make_slider(self, min_val: int, max_val: int, start: int, label_text: str) -> Tuple[QSlider, QLabel]:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(start)
        s.setSingleStep(1)
        l = QLabel(label_text)
        l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l.setMinimumWidth(52)
        return s, l

    def browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose sentence-centered dataset JSON", "", "JSON Files (*.json)")
        if path:
            self.dataset_edit.setText(path)
            self.load_dataset()

    def load_dataset(self) -> None:
        path = self.dataset_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Dataset not found", "Please choose a valid sentence-centered JSON file.")
            return
        try:
            self.model.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.log(f"Loaded dataset: {self.model.dataset_name}")
        self.log(f"Sentences: {len(self.model.sentences)} | unique atomic glyphs: {len(self.model.single_counts)}")
        self.recompute()

    def pmi_threshold(self) -> float:
        return self.pmi_slider.value() / 100.0

    def rigidity_penalty(self) -> float:
        return self.rig_slider.value() / 100.0

    def multi_penalty(self) -> float:
        return self.multi_slider.value() / 100.0

    def recompute(self) -> None:
        if not self.model.sentences:
            return
        pmi = self.pmi_threshold()
        rig = self.rigidity_penalty()
        multi = self.multi_penalty()
        self.pmi_value.setText(f"{pmi:.2f}")
        self.rig_value.setText(f"{rig:.2f}")
        self.multi_value.setText(f"{multi:.2f}")

        include_repeats = self.repeats_checkbox.isChecked()
        self.candidates = self.model.candidate_inventory(pmi, rig, multi, include_repeats)
        self.populate_table()
        self.update_plots()
        self.log(
            f"Refresh: PMI {pmi:.2f}, rigidity penalty {rig:.2f}, multi penalty {multi:.2f}, "
            f"repeated-pair override: {'on' if include_repeats else 'off'} | accepted candidates: {len(self.candidates)}"
        )
        if self.candidates:
            self.table.selectRow(0)
        else:
            self.detail_text.clear()
            self.examples_table.setRowCount(0)

    def populate_table(self) -> None:
        self.table.setRowCount(len(self.candidates))
        for r, cand in enumerate(self.candidates):
            values = [
                cand.chunk_display,
                str(cand.length),
                str(cand.count),
                f"{cand.avg_pmi:.3f}",
                f"{cand.min_pmi:.3f}",
                f"{cand.closure:.3f}",
                f"{cand.rigidity:.3f}",
                f"{cand.score:.3f}",
                cand.kind,
            ]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if c in (0, 8):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)

    def update_plots(self) -> None:
        # Bar chart: top accepted multi-glyph candidates by avg PMI.
        self.bar_canvas.fig.clear()
        ax = self.bar_canvas.fig.add_subplot(111)
        top = [c for c in self.candidates if c.length >= 2][:12]
        if top:
            labels = [c.chunk_display for c in top]
            vals = [c.avg_pmi for c in top]
            colors = ["tab:blue" if c.length == 2 else "tab:orange" for c in top]
            y = list(range(len(top)))[::-1]
            ax.barh(y, vals, color=colors)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Average PMI")
            ax.set_title("Top accepted multi-glyph candidates")
            for yi, cand, val in zip(y, top, vals):
                ax.text(val + 0.03, yi, f"score={cand.score:.2f}", va="center", fontsize=8)
            ax.grid(axis="x", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No accepted multi-glyph candidates", ha="center", va="center")
            ax.axis("off")
        self.bar_canvas.draw_idle()

        # Length pie.
        self.len_canvas.fig.clear()
        ax2 = self.len_canvas.fig.add_subplot(111)
        if self.candidates:
            counts = Counter(c.length for c in self.candidates)
            labels = [f"{k}-glyph" for k in sorted(counts)]
            vals = [counts[k] for k in sorted(counts)]
            ax2.pie(vals, labels=labels, autopct="%1.0f%%")
            ax2.set_title("Candidate pool by chunk length")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center")
            ax2.axis("off")
        self.len_canvas.draw_idle()

        # Position pie.
        self.pos_canvas.fig.clear()
        ax3 = self.pos_canvas.fig.add_subplot(111)
        if self.candidates:
            pos = Counter(self.position_class(c) for c in self.candidates)
            order = ["balanced", "middle-dominant", "start-dominant", "end-dominant"]
            labels = [k for k in order if pos.get(k, 0) > 0]
            vals = [pos[k] for k in labels]
            ax3.pie(vals, labels=labels, autopct="%1.0f%%")
            ax3.set_title("Where candidates tend to live")
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center")
            ax3.axis("off")
        self.pos_canvas.draw_idle()

    def position_class(self, cand: Candidate) -> str:
        total = max(1, cand.count)
        shares = {
            "start-dominant": cand.start_count / total,
            "middle-dominant": cand.mid_count / total,
            "end-dominant": cand.end_count / total,
        }
        k = max(shares, key=shares.get)
        return "balanced" if shares[k] < 0.60 else k

    def update_detail_panel(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.candidates):
            return
        cand = self.candidates[row]
        pos_class = self.position_class(cand)
        explanation = self.explain_candidate(cand)
        text = (
            f"Chunk: {cand.chunk_display}\n"
            f"Glyph IDs: {' '.join(cand.chunk_ids)}\n"
            f"Type: {cand.kind}\n"
            f"Length: {cand.length}\n"
            f"Count: {cand.count}\n"
            f"Average PMI: {cand.avg_pmi:.4f}\n"
            f"Minimum PMI: {cand.min_pmi:.4f}\n"
            f"Closure: {cand.closure:.4f}\n"
            f"Rigidity: {cand.rigidity:.4f}\n"
            f"Branching: {cand.branching:.4f}\n"
            f"Extension pressure: {cand.extension_pressure:.4f}\n"
            f"Repeated-sign bonus: {cand.repeated_bonus:.4f}\n"
            f"Score: {cand.score:.4f}\n"
            f"Position profile: start={cand.start_count}, mid={cand.mid_count}, end={cand.end_count} ({pos_class})\n\n"
            f"Why it survived:\n{explanation}"
        )
        self.detail_text.setPlainText(text)

        rows = self.model.example_rows(cand.chunk_ids)
        self.examples_table.setRowCount(len(rows))
        for r, (sid, glyph_sentence, panel_id) in enumerate(rows):
            vals = [str(sid), glyph_sentence, panel_id]
            for c, val in enumerate(vals):
                self.examples_table.setItem(r, c, QTableWidgetItem(val))

    def explain_candidate(self, cand: Candidate) -> str:
        pieces: List[str] = []
        if cand.length == 1:
            pieces.append("single glyphs are always kept as atomic inscription units")
        else:
            if cand.avg_pmi >= self.pmi_threshold():
                pieces.append("its internal neighbors are stickier than the current PMI threshold")
            if cand.closure >= 0.45:
                pieces.append("it behaves more like a self-contained chunk than a loose extension")
            if cand.kind == "repeated-sign candidate":
                pieces.append("the repeated-sign override is active, so this repeated pair is being tested as intentional")
            elif cand.repeated_bonus > 0:
                pieces.append("the repeated-sign pattern receives a modest intentional-compound bonus")
            if cand.length >= 3:
                pieces.append("despite the anti-3+ penalty, its score still cleared the longer-chunk hurdle")
            if cand.rigidity > 0.8:
                pieces.append("it is position-rigid, but other evidence still outweighed that penalty")
        if not pieces:
            pieces.append("it barely survived the current settings")
        return "- " + "\n- ".join(pieces)

    def export_json(self) -> None:
        if not self.candidates:
            QMessageBox.information(self, "Nothing to export", "No accepted candidates are available.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save accepted candidates JSON", "stage2_student_candidates_v3.json", "JSON Files (*.json)")
        if not path:
            return
        payload = {
            "dataset": self.model.dataset_name,
            "pmi_threshold": self.pmi_threshold(),
            "rigidity_penalty": self.rigidity_penalty(),
            "multi_glyph_penalty": self.multi_penalty(),
            "include_repeated_glyph_compounds": self.repeats_checkbox.isChecked(),
            "accepted_count": len(self.candidates),
            "rows": [
                {
                    "chunk_display": c.chunk_display,
                    "chunk_ids": list(c.chunk_ids),
                    "length": c.length,
                    "count": c.count,
                    "avg_pmi": c.avg_pmi,
                    "min_pmi": c.min_pmi,
                    "closure": c.closure,
                    "rigidity": c.rigidity,
                    "branching": c.branching,
                    "extension_pressure": c.extension_pressure,
                    "repeated_bonus": c.repeated_bonus,
                    "score": c.score,
                    "kind": c.kind,
                    "examples": c.examples,
                }
                for c in self.candidates
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.log(f"Saved JSON: {path}")

    def export_csv(self) -> None:
        if not self.candidates:
            QMessageBox.information(self, "Nothing to export", "No accepted candidates are available.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save accepted candidates CSV", "stage2_student_candidates_v3.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "chunk_display", "chunk_ids", "length", "count", "avg_pmi", "min_pmi",
                "closure", "rigidity", "branching", "extension_pressure",
                "repeated_bonus", "score", "kind", "examples"
            ])
            for c in self.candidates:
                writer.writerow([
                    c.chunk_display,
                    " ".join(c.chunk_ids),
                    c.length,
                    c.count,
                    f"{c.avg_pmi:.6f}",
                    f"{c.min_pmi:.6f}",
                    f"{c.closure:.6f}",
                    f"{c.rigidity:.6f}",
                    f"{c.branching:.6f}",
                    f"{c.extension_pressure:.6f}",
                    f"{c.repeated_bonus:.6f}",
                    f"{c.score:.6f}",
                    c.kind,
                    " ".join(map(str, c.examples)),
                ])
        self.log(f"Saved CSV: {path}")

    def log(self, text: str) -> None:
        self.status_box.append(text)


def main() -> int:
    app = QApplication(sys.argv)
    win = Stage2StudentWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
