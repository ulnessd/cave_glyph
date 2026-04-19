#!/usr/bin/env python3
"""
Stage 3 student-facing decipherment workbench (graph-oriented).

Purpose:
    - Load the restructured master dataset JSON
    - Load the Stage 2 student candidate inventory JSON
    - Build chunk <-> caption-term association evidence
    - Visualize the evidence as a bipartite graph
    - Let students assemble a provisional lexicon by choosing chunk-term links
    - Optionally suggest a one-to-one matching using weighted graph optimization

This is intentionally student-facing:
    - no hidden gold answers
    - no instructor reveals
    - evidence first, interpretation second
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

STOPWORDS = {
    "a", "an", "and", "are", "around", "as", "at", "be", "beside", "by", "clear",
    "depict", "depiction", "during", "for", "from",
    "has", "have", "in", "inside", "into", "is", "it", "its", "laid", "likely",
    "many", "may", "might", "near", "of", "on", "or", "out", "panel", "perhaps",
    "present", "returning", "scene", "seems", "several", "show", "shows",
    "some", "that", "the", "there", "this", "to", "under", "used",
    "view", "with", "within", "appears", "apparently", "first",
    "light", "open", "air", "image", "picture", "depicts", "showing", "survive", "surviving",
    "adjacent", "assumed", "nearby", "weathered", "find", "finds","occupying", "indicated", "apparently", "likely"
}

CONCEPT_ALIASES = {
    "person": "person",
    "people": "person",
    "individual": "person",
    "figure": "person",
    "figures": "person",
    "lone": "person",
    "solitary": "person",
    "single": "person",
    "tribe": "tribe",
    "group": "tribe",
    "assembled": "tribe",
    "gathered": "tribe",
    "attending": "tribe",
    "hunter": "hunter",
    "hunt": "hunter",
    "hunting": "hunter",
    "stalk": "hunter",
    "stalking": "hunter",
    "armed": "weapon",
    "weapon": "weapon",
    "weapons": "weapon",
    "spear": "weapon",
    "slain": "slain",
    "dead": "slain",
    "memorial": "slain",
    "water": "water",
    "river": "water",
    "bank": "water",
    "riverbank": "water",
    "fish": "fish",
    "fishing": "fish",
    "animal": "animal",
    "animals": "animal",
    "game": "animal",
    "food": "food",
    "prepared": "food",
    "stored": "food",
    "stores": "food",
    "storing": "food",
    "shelter": "shelter",
    "hearth": "fire",
    "fire": "fire",
    "tending": "fire",
    "maintained": "fire",
    "cave": "cave",
    "mountain": "mountain",
    "mountains": "mountain",
    "mountainous": "mountain",
    "country": "mountain",
    "sky": "sky",
    "day": "day",
    "daylight": "day",
    "night": "night",
    "nighttime": "night",
    "dawn": "dawn",
    "dusk": "dusk",
}

CLOSED_CONCEPTS = {
    "person",
    "tribe",
    "hunter",
    "weapon",
    "slain",
    "water",
    "fish",
    "animal",
    "food",
    "fire",
    "shelter",
    "cave",
    "mountain",
    "sky",
    "day",
    "night",
    "dawn",
    "dusk",
}

LATEX_TO_SYMBOL = {
    r"\vert": "|",
    r"\bigcirc": "○",
    r"\cdots": "···",
    r"\cap": "∩",
    r"\dagger": "†",
    r"\sim": "∼",
    r"\wedge": "∧",
    r"\Game": "⅁",
    r"\Finv": "Ⅎ",
    r"\rightthreetimes": "⋌",
}


BOILERPLATE_PATTERNS = [
    r"\bpanel appears to show\b",
    r"\bpicture appears to show\b",
    r"\bthis image seems to show\b",
    r"\bthis scene appears to show\b",
    r"\bthe panel may depict\b",
    r"\bthe panel seems to show\b",
    r"\bthere appear to be\b",
    r"\bappears to show\b",
    r"\bseems to show\b",
    r"\bmay depict\b",
    r"\bno adjacent cave art panel is assumed to survive\b",
    r"\bthis inscription is treated as a loose open air or weathered find\b",
]


@dataclass
class ChunkCandidate:
    display: str
    ids: List[str]
    length: int
    count: int
    score: float
    avg_pmi: float
    min_pmi: float
    kind: str
    examples: List[int]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelRecord:
    panel_id: str
    caption: str
    sentence_numbers: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


@dataclass
class HypothesisRecord:
    chunk_display: str
    chunk_ids: List[str]
    proposed_gloss: str
    evidence_score: float
    source: str = "manual"
    notes: str = ""


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sentence_display(sentence: Dict[str, Any]) -> str:
    text = sentence["student_view"].get("glyph_sentence_latex", "")
    text = text.replace("\\quad", " ")
    return " ".join(text.split())


def latex_word_to_symbols(word: str) -> str:
    out = word
    for latex, sym in sorted(LATEX_TO_SYMBOL.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(latex, sym)
    out = out.replace("{", "").replace("}", "")
    return " ".join(out.split())


def segmented_symbol_display(sentence: Dict[str, Any]) -> str:
    words = sentence.get("student_view", {}).get("segmented_word_glyphs_latex", [])
    if not words:
        words = sentence.get("student_view", {}).get("atomic_glyph_stream_latex", [])
    return "   ".join(latex_word_to_symbols(w) for w in words)


def normalize_caption(text: str, strip_boilerplate: bool, normalize_concepts: bool = True) -> List[str]:
    text = text.lower()
    if strip_boilerplate:
        for patt in BOILERPLATE_PATTERNS:
            text = re.sub(patt, " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    out: List[str] = []
    for tok in text.split():
        if not tok or tok in STOPWORDS or len(tok) < 2:
            continue

        if normalize_concepts:
            mapped = CONCEPT_ALIASES.get(tok)
            if not mapped:
                continue
            if mapped not in CLOSED_CONCEPTS:
                continue
            tok = mapped

        out.append(tok)

    return out

def cosine_from_dicts(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    num = sum(a[k] * b[k] for k in keys)
    den1 = math.sqrt(sum(v * v for v in a.values()))
    den2 = math.sqrt(sum(v * v for v in b.values()))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def set_table_headers(table: QTableWidget, headers: List[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)


class GraphCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.fig)


class Stage3GraphWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stage 3 Decipherment Workbench — Graph View v4")
        self.resize(1480, 940)

        self.dataset_path: Optional[str] = None
        self.inventory_path: Optional[str] = None
        self.dataset: Optional[Dict[str, Any]] = None
        self.sentences: List[Dict[str, Any]] = []
        self.sent_by_num: Dict[int, Dict[str, Any]] = {}
        self.panels: Dict[str, PanelRecord] = {}
        self.candidates: List[ChunkCandidate] = []
        self.filtered_candidates: List[ChunkCandidate] = []
        self.chunk_panel_sets: Dict[str, Set[str]] = {}
        self.chunk_term_weights: Dict[str, Dict[str, float]] = {}
        self.term_chunk_weights: Dict[str, Dict[str, float]] = {}
        self.term_panel_sets: Dict[str, Set[str]] = {}
        self.working_lexicon: Dict[str, HypothesisRecord] = {}
        self.matching_suggestions: List[Tuple[str, str, float]] = []
        self.current_chunk: Optional[ChunkCandidate] = None
        self.current_term: Optional[str] = None
        self.current_hyp_chunk: Optional[str] = None
        self.chunk_choice_options: Dict[str, List[Tuple[str, float]]] = {}
        self.current_assignment: Dict[str, str] = {}
        self.hyp_graph_positions: Dict[str, Tuple[float, float]] = {}
        self.hyp_graph_nodes: Dict[str, str] = {}
        self._updating_hyp_combo = False
        self.hyp_clean_layout = False

        self._build_ui()
        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        open_dataset = QAction("Open master dataset…", self)
        open_dataset.triggered.connect(self.browse_dataset)
        menu.addAction(open_dataset)
        open_inventory = QAction("Open Stage 2 candidates…", self)
        open_inventory.triggered.connect(self.browse_inventory)
        menu.addAction(open_inventory)
        menu.addSeparator()
        save_lex = QAction("Save working lexicon…", self)
        save_lex.triggered.connect(self.save_working_lexicon)
        menu.addAction(save_lex)
        export_assoc = QAction("Export association report…", self)
        export_assoc.triggered.connect(self.export_association_report)
        menu.addAction(export_assoc)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        # ----- Left panel -----
        left = QWidget()
        left.setMinimumWidth(330)
        left_layout = QVBoxLayout(left)

        src_box = QGroupBox("Data Sources")
        src_form = QFormLayout(src_box)
        self.dataset_edit = QLineEdit()
        self.inventory_edit = QLineEdit()
        ds_row = QHBoxLayout(); ds_row.addWidget(self.dataset_edit); ds_b = QPushButton("Browse…"); ds_b.clicked.connect(self.browse_dataset); ds_row.addWidget(ds_b)
        inv_row = QHBoxLayout(); inv_row.addWidget(self.inventory_edit); inv_b = QPushButton("Browse…"); inv_b.clicked.connect(self.browse_inventory); inv_row.addWidget(inv_b)
        src_form.addRow("Master dataset JSON", self._wrap(ds_row))
        src_form.addRow("Stage 2 candidates JSON", self._wrap(inv_row))
        left_layout.addWidget(src_box)

        filt_box = QGroupBox("Analysis Filters")
        filt_form = QFormLayout(filt_box)
        self.min_score_spin = QDoubleSpinBox(); self.min_score_spin.setRange(-999.0, 999.0); self.min_score_spin.setDecimals(2); self.min_score_spin.setValue(-999.0)
        self.min_term_freq_spin = QSpinBox(); self.min_term_freq_spin.setRange(1, 20); self.min_term_freq_spin.setValue(2)
        self.accepted_only_check = QCheckBox("Accepted Stage 2 candidates only"); self.accepted_only_check.setChecked(True)
        self.atomic_check = QCheckBox("Include atomic glyphs"); self.atomic_check.setChecked(True)
        self.lexical_check = QCheckBox("Include likely lexical pairs"); self.lexical_check.setChecked(True)
        self.repeated_check = QCheckBox("Include repeated-sign candidates"); self.repeated_check.setChecked(True)
        self.formulaic_check = QCheckBox("Include formulaic / unresolved chunks"); self.formulaic_check.setChecked(False)
        self.strip_boilerplate_check = QCheckBox("Strip caption boilerplate"); self.strip_boilerplate_check.setChecked(True)
        self.normalize_concepts_check = QCheckBox("Use normalized concept nodes"); self.normalize_concepts_check.setChecked(True)
        filt_form.addRow("Minimum Stage 2 score", self.min_score_spin)
        filt_form.addRow("Min panel-term frequency", self.min_term_freq_spin)
        filt_form.addRow("", self.accepted_only_check)
        filt_form.addRow("", self.atomic_check)
        filt_form.addRow("", self.lexical_check)
        filt_form.addRow("", self.repeated_check)
        filt_form.addRow("", self.formulaic_check)
        filt_form.addRow("", self.strip_boilerplate_check)
        filt_form.addRow("", self.normalize_concepts_check)
        left_layout.addWidget(filt_box)

        act_box = QGroupBox("Actions")
        act_layout = QVBoxLayout(act_box)
        self.load_btn = QPushButton("Load dataset + candidates"); self.load_btn.clicked.connect(self.load_inputs)
        self.run_btn = QPushButton("Run semantic grounding analysis"); self.run_btn.clicked.connect(self.run_analysis)
        self.suggest_btn = QPushButton("Suggest one-to-one matching"); self.suggest_btn.clicked.connect(self.suggest_matching)
        self.save_lex_btn = QPushButton("Save working lexicon JSON"); self.save_lex_btn.clicked.connect(self.save_working_lexicon)
        self.export_btn = QPushButton("Export association report JSON"); self.export_btn.clicked.connect(self.export_association_report)
        for b in (self.load_btn, self.run_btn, self.suggest_btn, self.save_lex_btn, self.export_btn):
            act_layout.addWidget(b)
        left_layout.addWidget(act_box)

        self.summary_box = QTextEdit(); self.summary_box.setReadOnly(True); self.summary_box.setMinimumHeight(220)
        left_layout.addWidget(self.summary_box)
        left_layout.addStretch(1)

        # ----- Tabs -----
        self.tabs = QTabWidget()
        self._build_panel_tab()
        self._build_graph_tab()
        self._build_assoc_tab()
        self._build_hyp_tab()
        self._build_sentence_challenge_tab()

        outer.addWidget(left, 0)
        outer.addWidget(self.tabs, 1)

    def _wrap(self, layout: QHBoxLayout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    def _build_panel_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget(); l = QVBoxLayout(left)
        self.panel_filter_edit = QLineEdit(); self.panel_filter_edit.setPlaceholderText("Filter panels or captions…"); self.panel_filter_edit.textChanged.connect(self.refresh_panel_list)
        self.panel_list = QListWidget(); self.panel_list.currentRowChanged.connect(self.update_panel_detail)
        l.addWidget(self.panel_filter_edit); l.addWidget(self.panel_list)

        right = QWidget(); r = QVBoxLayout(right)
        self.panel_detail = QTextEdit(); self.panel_detail.setReadOnly(True)
        self.panel_sentence_table = QTableWidget(); set_table_headers(self.panel_sentence_table, ["Sentence #", "Glyph sentence", "Active chunks"])
        r.addWidget(self.panel_detail, 1)
        r.addWidget(self.panel_sentence_table, 2)

        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Panel Explorer")

    def _build_graph_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget(); l = QVBoxLayout(left)
        self.graph_mode_combo = QComboBox(); self.graph_mode_combo.addItems(["Chunk-centered", "Term-centered", "Global top edges"]); self.graph_mode_combo.currentIndexChanged.connect(self.refresh_graph_sources)
        self.graph_filter_edit = QLineEdit(); self.graph_filter_edit.setPlaceholderText("Filter chunks or terms…"); self.graph_filter_edit.textChanged.connect(self.refresh_graph_sources)
        self.graph_source_list = QListWidget(); self.graph_source_list.currentRowChanged.connect(self.update_graph_view)
        l.addWidget(QLabel("Graph mode")); l.addWidget(self.graph_mode_combo)
        l.addWidget(self.graph_filter_edit); l.addWidget(self.graph_source_list)

        middle = QWidget(); m = QVBoxLayout(middle)
        self.graph_canvas = GraphCanvas()
        self.graph_detail = QTextEdit(); self.graph_detail.setReadOnly(True); self.graph_detail.setMinimumHeight(140)
        m.addWidget(self.graph_canvas, 3)
        m.addWidget(self.graph_detail, 1)

        right = QWidget(); r = QVBoxLayout(right)
        self.edge_table = QTableWidget(); set_table_headers(self.edge_table, ["Neighbor", "Weight", "Panels", "Type"])
        self.edge_table.currentCellChanged.connect(self.edge_row_changed)
        assign_box = QGroupBox("Adopt hypothesis from selected edge")
        af = QFormLayout(assign_box)
        self.selected_chunk_label = QLabel("(none)")
        self.selected_term_label = QLabel("(none)")
        self.graph_custom_gloss_edit = QLineEdit()
        self.graph_custom_gloss_edit.setPlaceholderText("Optional override gloss")
        self.graph_add_hyp_btn = QPushButton("Add / update hypothesis")
        # Keep graph-tab adoption inert for now; use the Hypothesis Builder tab for actual updates.
        self.graph_add_hyp_btn.clicked.connect(lambda: None)
        af.addRow("Chunk", self.selected_chunk_label)
        af.addRow("Term", self.selected_term_label)
        af.addRow("Custom gloss", self.graph_custom_gloss_edit)
        af.addRow("", self.graph_add_hyp_btn)
        r.addWidget(self.edge_table, 2)
        r.addWidget(assign_box, 1)

        splitter.addWidget(left); splitter.addWidget(middle); splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Graph Workbench")

    def _build_assoc_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget(); l = QVBoxLayout(left)
        self.term_filter_edit = QLineEdit(); self.term_filter_edit.setPlaceholderText("Filter concept nodes…"); self.term_filter_edit.textChanged.connect(self.refresh_term_tables)
        self.term_table = QTableWidget(); set_table_headers(self.term_table, ["Concept", "Panels", "Best chunk", "Best score"])
        self.term_table.currentCellChanged.connect(self.update_term_chunk_table)
        l.addWidget(self.term_filter_edit); l.addWidget(self.term_table)

        right = QWidget(); r = QVBoxLayout(right)
        self.term_chunk_table = QTableWidget(); set_table_headers(self.term_chunk_table, ["Chunk", "Kind", "Association", "Panels"])
        r.addWidget(self.term_chunk_table)

        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Association Matrix")

    def _build_hyp_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.score_label = QLabel("No hypotheses yet.")
        self.adopt_match_btn = QPushButton("Adopt current matching suggestions")
        self.adopt_match_btn.clicked.connect(self.adopt_matching_suggestions)

        self.cleanup_layout_btn = QPushButton("Graph clean-up")
        self.cleanup_layout_btn.clicked.connect(self.cleanup_hypothesis_layout)

        self.clear_hyp_btn = QPushButton("Reset to top edges")
        self.clear_hyp_btn.clicked.connect(self.reset_hypotheses_to_top_edges)

        self.remove_hyp_btn = QPushButton("Remove selected hypothesis")
        self.remove_hyp_btn.clicked.connect(self.remove_selected_hypothesis)

        top.addWidget(self.score_label, 1)
        top.addWidget(self.adopt_match_btn)
        top.addWidget(self.cleanup_layout_btn)
        top.addWidget(self.clear_hyp_btn)
        top.addWidget(self.remove_hyp_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget(); l = QVBoxLayout(left)
        self.hyp_graph_canvas = GraphCanvas()
        self.hyp_graph_canvas.mpl_connect("button_press_event", self.on_hyp_graph_click)
        self.hyp_graph_detail = QTextEdit(); self.hyp_graph_detail.setReadOnly(True); self.hyp_graph_detail.setMinimumHeight(130)
        l.addWidget(self.hyp_graph_canvas, 3)
        l.addWidget(self.hyp_graph_detail, 1)

        right = QWidget(); r = QVBoxLayout(right)
        ctrl_box = QGroupBox("Selected chunk")
        ctrl_form = QFormLayout(ctrl_box)
        self.hyp_selected_chunk_label = QLabel("(click a chunk node)")
        self.hyp_term_combo = QComboBox()
        self.hyp_term_combo.currentIndexChanged.connect(self.hyp_term_changed)
        self.custom_gloss_edit = QLineEdit(); self.custom_gloss_edit.setPlaceholderText("Optional custom gloss label")
        self.add_hyp_btn = QPushButton("Apply selected edge")
        self.add_hyp_btn.clicked.connect(self.add_hypothesis_from_graph_choice)
        ctrl_form.addRow("Chunk", self.hyp_selected_chunk_label)
        ctrl_form.addRow("Candidate term", self.hyp_term_combo)
        ctrl_form.addRow("Custom gloss", self.custom_gloss_edit)
        ctrl_form.addRow("", self.add_hyp_btn)
        r.addWidget(ctrl_box)

        self.hyp_assignment_table = QTableWidget(); set_table_headers(self.hyp_assignment_table, ["Chunk", "Chosen term", "Edge wt", "Rank", "Gloss"])
        self.hyp_assignment_table.currentCellChanged.connect(self.hyp_assignment_row_changed)
        self.match_table = QTableWidget(); set_table_headers(self.match_table, ["Chunk", "Suggested term", "Weight"])
        r.addWidget(QLabel("Current graph hypotheses"))
        r.addWidget(self.hyp_assignment_table, 2)
        r.addWidget(QLabel("Current one-to-one matching suggestion"))
        r.addWidget(self.match_table, 1)

        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Hypothesis Builder")

    def _build_sentence_challenge_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        intro = QLabel(
            "Use your best chunk-to-word dictionary from Tab 4. Select a random inscription, "
            "write your own interpretation, then reveal the original English sentence and reflect on the differences."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        btn_row = QHBoxLayout()
        self.challenge_random_btn = QPushButton("Select random sentence")
        self.challenge_random_btn.clicked.connect(self.select_random_sentence)
        self.challenge_reveal_btn = QPushButton("Reveal true sentence")
        self.challenge_reveal_btn.clicked.connect(self.reveal_true_sentence)
        self.challenge_reveal_btn.setEnabled(False)
        btn_row.addWidget(self.challenge_random_btn)
        btn_row.addWidget(self.challenge_reveal_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.challenge_sentence_num_label = QLabel("Sentence: (none)")
        layout.addWidget(self.challenge_sentence_num_label)

        self.challenge_glyph_label = QLabel("Load the dataset, then choose a random sentence.")
        self.challenge_glyph_label.setWordWrap(True)
        self.challenge_glyph_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.challenge_glyph_label.setStyleSheet("font-size: 22px; padding: 10px; border: 1px solid #999;")
        layout.addWidget(self.challenge_glyph_label)

        layout.addWidget(QLabel("Your interpretation"))
        self.challenge_student_interp = QPlainTextEdit()
        self.challenge_student_interp.setPlaceholderText("Write your best English interpretation here before revealing the original sentence.")
        self.challenge_student_interp.setMinimumHeight(120)
        layout.addWidget(self.challenge_student_interp)

        layout.addWidget(QLabel("Revealed original sentence and note"))
        self.challenge_reveal_text = QTextEdit()
        self.challenge_reveal_text.setReadOnly(True)
        self.challenge_reveal_text.setMinimumHeight(200)
        layout.addWidget(self.challenge_reveal_text, 1)

        self.challenge_current_sentence: Optional[Dict[str, Any]] = None

        self.tabs.addTab(tab, "Sentence Challenge")

    # ---------- File actions ----------
    def browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select master dataset JSON", "", "JSON files (*.json)")
        if path:
            self.dataset_edit.setText(path)
            self.dataset_path = path

    def browse_inventory(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Stage 2 candidates JSON", "", "JSON files (*.json)")
        if path:
            self.inventory_edit.setText(path)
            self.inventory_path = path

    def load_inputs(self) -> None:
        self.dataset_path = self.dataset_edit.text().strip()
        self.inventory_path = self.inventory_edit.text().strip()
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            QMessageBox.warning(self, "Missing dataset", "Please choose a valid master dataset JSON file.")
            return
        if not self.inventory_path or not os.path.exists(self.inventory_path):
            QMessageBox.warning(self, "Missing Stage 2 candidates", "Please choose a valid Stage 2 candidates JSON file.")
            return
        try:
            self.dataset = read_json(self.dataset_path)
            self.sentences = self.dataset.get("sentences", [])
            self.sent_by_num = {s["sentence_number"]: s for s in self.sentences}
            self._build_panels()
            self._load_candidates()
            self.run_analysis()
            self.summary_box.append("Loaded dataset and candidate inventory.")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not load inputs:\n\n{exc}")

    def _build_panels(self) -> None:
        self.panels = {}
        for s in self.sentences:
            sv = s["student_view"]
            pid = sv.get("linked_panel_id")
            caption = sv.get("linked_panel_caption")
            if pid and caption:
                rec = self.panels.setdefault(pid, PanelRecord(panel_id=pid, caption=caption))
                rec.sentence_numbers.append(s["sentence_number"])

    def _load_candidates(self) -> None:
        inv = read_json(self.inventory_path)
        self.candidates = []
        for row in inv.get("rows", []):
            examples = []
            seen = set()
            for ex in row.get("examples", []):
                if ex not in seen:
                    seen.add(ex)
                    examples.append(ex)
            self.candidates.append(
                ChunkCandidate(
                    display=row.get("chunk_display", ""),
                    ids=row.get("chunk_ids", []),
                    length=int(row.get("length", 0)),
                    count=int(row.get("count", 0)),
                    score=float(row.get("score", 0.0)),
                    avg_pmi=float(row.get("avg_pmi", 0.0)),
                    min_pmi=float(row.get("min_pmi", 0.0)),
                    kind=row.get("kind", "candidate"),
                    examples=examples,
                    raw=row,
                )
            )

    # ---------- Analysis ----------
    def _candidate_allowed(self, c: ChunkCandidate) -> bool:
        if self.accepted_only_check.isChecked() and not c.raw.get("accepted", True):
            return False
        if c.score < self.min_score_spin.value():
            return False
        k = c.kind.lower()
        if "atomic" in k and not self.atomic_check.isChecked():
            return False
        if ("lexical" in k or "pair" in k) and not self.lexical_check.isChecked():
            return False
        if "repeated" in k and not self.repeated_check.isChecked():
            return False
        if ("formulaic" in k or "unresolved" in k or "weak" in k) and not self.formulaic_check.isChecked():
            return False
        return True

    def _initialize_hypothesis_state(self) -> None:
        self.chunk_choice_options = {}
        self.current_assignment = {}
        self.working_lexicon = {}
        for c in self.filtered_candidates:
            opts = sorted(self.chunk_term_weights.get(c.display, {}).items(), key=lambda x: (-x[1], x[0]))[:6]
            self.chunk_choice_options[c.display] = opts
            if opts:
                term, wt = opts[0]
                self.current_assignment[c.display] = term
                self.working_lexicon[c.display] = HypothesisRecord(
                    chunk_display=c.display,
                    chunk_ids=c.ids,
                    proposed_gloss=term,
                    evidence_score=wt,
                    source="graph-top",
                    notes="",
                )

    def run_analysis(self) -> None:
        if not self.sentences or not self.candidates:
            return
        self.filtered_candidates = [c for c in self.candidates if self._candidate_allowed(c)]
        self.chunk_panel_sets = {}
        self.chunk_term_weights = {}
        self.term_chunk_weights = defaultdict(dict)
        self.term_panel_sets = defaultdict(set)

        strip_boilerplate = self.strip_boilerplate_check.isChecked()
        normalize_concepts = self.normalize_concepts_check.isChecked()
        min_term_freq = self.min_term_freq_spin.value()

        # Tokenize captions.
        for rec in self.panels.values():
            rec.tokens = normalize_caption(rec.caption, strip_boilerplate, normalize_concepts)
            for tok in set(rec.tokens):
                self.term_panel_sets[tok].add(rec.panel_id)

        # Precompute sentence->panel mapping.
        sentence_to_panel: Dict[int, Optional[str]] = {}
        for s in self.sentences:
            sentence_to_panel[s["sentence_number"]] = s["student_view"].get("linked_panel_id")

        total_panels = max(1, len(self.panels))
        for c in self.filtered_candidates:
            panel_ids = {sentence_to_panel[n] for n in c.examples if sentence_to_panel.get(n)}
            panel_ids.discard(None)
            panel_ids = set(panel_ids)
            self.chunk_panel_sets[c.display] = panel_ids
            weights: Dict[str, float] = {}
            if not panel_ids:
                self.chunk_term_weights[c.display] = weights
                continue
            panel_counter = Counter()
            for pid in panel_ids:
                rec = self.panels.get(pid)
                if not rec:
                    continue
                for tok in set(rec.tokens):
                    panel_counter[tok] += 1
            for term, overlap in panel_counter.items():
                if len(self.term_panel_sets[term]) < min_term_freq:
                    continue
                c_docs = len(panel_ids)
                t_docs = len(self.term_panel_sets[term])
                # Blend cosine-ish overlap with positive PMI.
                overlap_norm = overlap / math.sqrt(max(1, c_docs) * max(1, t_docs))
                pmi = math.log((overlap * total_panels) / max(1.0, c_docs * t_docs)) if overlap > 0 else 0.0
                assoc = overlap_norm + max(0.0, pmi) * 0.35
                if assoc > 0:
                    weights[term] = assoc
                    self.term_chunk_weights[term][c.display] = assoc
            self.chunk_term_weights[c.display] = weights

        self._initialize_hypothesis_state()
        term_label = "concept nodes" if normalize_concepts else "caption terms"
        self.summary_box.append(
            f"Semantic analysis complete. {len(self.filtered_candidates)} active chunks; {len(self.term_chunk_weights)} {term_label}."
        )
        self.refresh_panel_list()
        self.refresh_graph_sources()
        self.refresh_term_tables()
        self.refresh_hypothesis_tables()
        if self.filtered_candidates:
            self.select_hyp_chunk(self.filtered_candidates[0].display)
        self.suggest_matching()

    # ---------- Panel tab ----------
    def refresh_panel_list(self) -> None:
        filt = self.panel_filter_edit.text().strip().lower()
        self.panel_list.clear()
        for pid, rec in sorted(self.panels.items()):
            label = f"{pid}: {rec.caption[:70]}"
            if filt and filt not in label.lower():
                continue
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self.panel_list.addItem(item)
        if self.panel_list.count() > 0 and self.panel_list.currentRow() < 0:
            self.panel_list.setCurrentRow(0)

    def update_panel_detail(self) -> None:
        item = self.panel_list.currentItem()
        if not item:
            self.panel_detail.clear(); self.panel_sentence_table.setRowCount(0); return
        pid = item.data(Qt.ItemDataRole.UserRole)
        rec = self.panels[pid]
        self.panel_detail.setPlainText(
            f"Panel: {pid}\n\nCaption:\n{rec.caption}\n\nLinked inscriptions: {', '.join(str(n) for n in rec.sentence_numbers)}"
        )
        rows: List[Tuple[int, str, str]] = []
        active_chunks = []
        for sn in rec.sentence_numbers:
            s = self.sent_by_num.get(sn)
            if not s:
                continue
            glyph = sentence_display(s)
            present = []
            atomic = " ".join(s["student_view"].get("atomic_glyph_stream_latex", []))
            for c in self.filtered_candidates:
                disp = c.display.replace(" ", "")
                atom = atomic.replace(" ", "")
                if disp and disp in atom:
                    present.append(c.display)
            rows.append((sn, glyph, ", ".join(present[:8])))
        self.panel_sentence_table.setRowCount(len(rows))
        for r, (sn, glyph, present) in enumerate(rows):
            self.panel_sentence_table.setItem(r, 0, QTableWidgetItem(str(sn)))
            self.panel_sentence_table.setItem(r, 1, QTableWidgetItem(glyph))
            self.panel_sentence_table.setItem(r, 2, QTableWidgetItem(present))

    # ---------- Graph tab ----------
    def refresh_graph_sources(self) -> None:
        mode = self.graph_mode_combo.currentText()
        filt = self.graph_filter_edit.text().strip().lower()
        self.graph_source_list.clear()
        if mode == "Chunk-centered":
            for c in self.filtered_candidates:
                label = f"{c.display} [{c.kind}]"
                if filt and filt not in label.lower() and filt not in " ".join(c.ids).lower():
                    continue
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, ("chunk", c.display))
                self.graph_source_list.addItem(item)
        elif mode == "Term-centered":
            for term, chunkmap in sorted(self.term_chunk_weights.items()):
                if filt and filt not in term.lower():
                    continue
                item = QListWidgetItem(f"{term} ({len(chunkmap)})")
                item.setData(Qt.ItemDataRole.UserRole, ("term", term))
                self.graph_source_list.addItem(item)
        else:
            item = QListWidgetItem("Top global chunk–term edges")
            item.setData(Qt.ItemDataRole.UserRole, ("global", "top"))
            self.graph_source_list.addItem(item)
        if self.graph_source_list.count() > 0 and self.graph_source_list.currentRow() < 0:
            self.graph_source_list.setCurrentRow(0)

    def update_graph_view(self) -> None:
        item = self.graph_source_list.currentItem()
        if not item:
            return
        kind, key = item.data(Qt.ItemDataRole.UserRole)
        if kind == "chunk":
            cand = next((c for c in self.filtered_candidates if c.display == key), None)
            if cand:
                self.draw_chunk_graph(cand)
        elif kind == "term":
            self.draw_term_graph(key)
        else:
            self.draw_global_graph()

    def draw_chunk_graph(self, cand: ChunkCandidate) -> None:
        self.current_chunk = cand
        weights = sorted(self.chunk_term_weights.get(cand.display, {}).items(), key=lambda x: (-x[1], x[0]))[:8]
        G = nx.Graph()
        center = f"C::{cand.display}"
        G.add_node(center, bipartite=0, label=cand.display, ntype="chunk")
        pos = {center: (0.15, 0.5)}
        for idx, (term, wt) in enumerate(weights):
            node = f"T::{term}"
            G.add_node(node, bipartite=1, label=term, ntype="term")
            y = 0.9 - idx * (0.8 / max(1, len(weights) - 1 if len(weights) > 1 else 1))
            pos[node] = (0.85, y)
            G.add_edge(center, node, weight=wt)
        self._draw_graph(G, pos, highlight=center)
        neighbor_type = "concept" if self.normalize_concepts_check.isChecked() else "caption term"
        self._fill_edge_table([(term, wt, len(self.term_panel_sets.get(term, set()) & self.chunk_panel_sets.get(cand.display, set())), neighbor_type) for term, wt in weights], chunk=cand.display)
        detail = [
            f"Chunk: {cand.display}",
            f"Glyph IDs: {' '.join(cand.ids)}",
            f"Kind: {cand.kind}",
            f"Count: {cand.count}",
            f"Score: {cand.score:.4f}",
            f"Panels: {', '.join(sorted(self.chunk_panel_sets.get(cand.display, set())))}",
        ]
        if weights:
            detail.append("\nStrongest associated terms:")
            detail.extend([f"- {t}: {w:.4f}" for t, w in weights])
        self.graph_detail.setPlainText("\n".join(detail))
        self.selected_chunk_label.setText(cand.display)
        self.selected_term_label.setText("(select edge)")

    def draw_term_graph(self, term: str) -> None:
        self.current_term = term
        chunk_items = sorted(self.term_chunk_weights.get(term, {}).items(), key=lambda x: (-x[1], x[0]))[:8]
        G = nx.Graph()
        center = f"T::{term}"
        G.add_node(center, bipartite=1, label=term, ntype="term")
        pos = {center: (0.85, 0.5)}
        for idx, (chunk, wt) in enumerate(chunk_items):
            node = f"C::{chunk}"
            G.add_node(node, bipartite=0, label=chunk, ntype="chunk")
            y = 0.9 - idx * (0.8 / max(1, len(chunk_items) - 1 if len(chunk_items) > 1 else 1))
            pos[node] = (0.15, y)
            G.add_edge(node, center, weight=wt)
        self._draw_graph(G, pos, highlight=center)
        self._fill_edge_table([(chunk, wt, len(self.chunk_panel_sets.get(chunk, set()) & self.term_panel_sets.get(term, set())), self._kind_for_chunk(chunk)) for chunk, wt in chunk_items], term=term)
        node_label = "Concept" if self.normalize_concepts_check.isChecked() else "Caption term"
        detail = [
            f"{node_label}: {term}",
            f"Panels containing node: {', '.join(sorted(self.term_panel_sets.get(term, set())))}",
            f"Strongest chunks:"
        ] + [f"- {ch}: {wt:.4f}" for ch, wt in chunk_items]
        self.graph_detail.setPlainText("\n".join(detail))
        self.selected_term_label.setText(term)
        self.selected_chunk_label.setText("(select edge)")

    def draw_global_graph(self) -> None:
        edges = []
        for term, cmap in self.term_chunk_weights.items():
            for chunk, wt in cmap.items():
                edges.append((chunk, term, wt))
        edges.sort(key=lambda x: (-x[2], x[0], x[1]))
        edges = edges[:12]
        G = nx.Graph(); pos = {}
        chunk_nodes = sorted({c for c, _, _ in edges}); term_nodes = sorted({t for _, t, _ in edges})
        for i, ch in enumerate(chunk_nodes):
            node = f"C::{ch}"; G.add_node(node, bipartite=0, label=ch, ntype="chunk"); pos[node] = (0.15, 0.9 - i * (0.8 / max(1, len(chunk_nodes)-1 if len(chunk_nodes)>1 else 1)))
        for i, term in enumerate(term_nodes):
            node = f"T::{term}"; G.add_node(node, bipartite=1, label=term, ntype="term"); pos[node] = (0.85, 0.9 - i * (0.8 / max(1, len(term_nodes)-1 if len(term_nodes)>1 else 1)))
        for ch, term, wt in edges:
            G.add_edge(f"C::{ch}", f"T::{term}", weight=wt)
        self._draw_graph(G, pos, highlight=None)
        self._fill_edge_table([(f"{ch} ↔ {term}", wt, 0, "edge") for ch, term, wt in edges])
        self.graph_detail.setPlainText("Global view of the strongest chunk–term associations under the current filters.")
        self.selected_chunk_label.setText("(select edge)")
        self.selected_term_label.setText("(select edge)")

    def _draw_graph(self, G: nx.Graph, pos: Dict[str, Tuple[float, float]], highlight: Optional[str]) -> None:
        self.graph_canvas.fig.clear()
        ax = self.graph_canvas.fig.add_subplot(111)
        ax.set_axis_off()
        if G.number_of_nodes() == 0:
            self.graph_canvas.draw_idle(); return
        chunk_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "chunk"]
        term_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "term"]
        node_colors = []
        for n in G.nodes():
            if n == highlight:
                node_colors.append("#f59e0b")
            elif n.startswith("C::"):
                node_colors.append("#2563eb")
            else:
                node_colors.append("#059669")
        widths = [1.5 + 2.5 * G.edges[e].get("weight", 0.0) for e in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()), node_size=1500, node_color=node_colors, alpha=0.95)
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=9, font_color="white")
        self.graph_canvas.draw_idle()

    def _fill_edge_table(self, rows: List[Tuple[str, float, int, str]], chunk: Optional[str] = None, term: Optional[str] = None) -> None:
        self.edge_table.setRowCount(len(rows))
        for r, (name, wt, panels, typ) in enumerate(rows):
            self.edge_table.setItem(r, 0, QTableWidgetItem(str(name)))
            self.edge_table.setItem(r, 1, QTableWidgetItem(f"{wt:.4f}"))
            self.edge_table.setItem(r, 2, QTableWidgetItem(str(panels)))
            self.edge_table.setItem(r, 3, QTableWidgetItem(typ))
            payload = {"chunk": chunk, "term": term, "neighbor": name, "weight": wt, "type": typ}
            self.edge_table.item(r, 0).setData(Qt.ItemDataRole.UserRole, payload)
        if self.edge_table.rowCount() > 0:
            self.edge_table.selectRow(0)
            self.edge_row_changed(0, 0, -1, -1)

    def edge_row_changed(self, currentRow: int, currentColumn: int, previousRow: int, previousColumn: int) -> None:
        item = self.edge_table.item(currentRow, 0)
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        chunk = data.get("chunk")
        term = data.get("term")
        neigh = data.get("neighbor")
        if chunk and not term:
            self.selected_chunk_label.setText(chunk)
            self.selected_term_label.setText(neigh)
        elif term and not chunk:
            self.selected_term_label.setText(term)
            self.selected_chunk_label.setText(neigh)
        else:
            # Global row "chunk ↔ term"
            if "↔" in neigh:
                ch, tm = [p.strip() for p in neigh.split("↔", 1)]
                self.selected_chunk_label.setText(ch)
                self.selected_term_label.setText(tm)

    # ---------- Associations tab ----------
    def refresh_term_tables(self) -> None:
        filt = self.term_filter_edit.text().strip().lower()
        rows = []
        for term, cmap in self.term_chunk_weights.items():
            if filt and filt not in term.lower():
                continue
            if not cmap:
                continue
            best_chunk, best_score = max(cmap.items(), key=lambda x: x[1])
            rows.append((term, len(self.term_panel_sets.get(term, set())), best_chunk, best_score))
        rows.sort(key=lambda x: (-x[3], x[0]))
        self.term_table.setRowCount(len(rows))
        for r, (term, panels, best_chunk, best_score) in enumerate(rows):
            self.term_table.setItem(r, 0, QTableWidgetItem(term))
            self.term_table.setItem(r, 1, QTableWidgetItem(str(panels)))
            self.term_table.setItem(r, 2, QTableWidgetItem(best_chunk))
            self.term_table.setItem(r, 3, QTableWidgetItem(f"{best_score:.4f}"))
        if self.term_table.rowCount() > 0:
            self.term_table.selectRow(0)
            self.update_term_chunk_table(0, 0, -1, -1)

    def update_term_chunk_table(self, currentRow: int, currentColumn: int, previousRow: int, previousColumn: int) -> None:
        item = self.term_table.item(currentRow, 0)
        if not item:
            self.term_chunk_table.setRowCount(0); return
        term = item.text()
        cmap = self.term_chunk_weights.get(term, {})
        rows = sorted(cmap.items(), key=lambda x: (-x[1], x[0]))
        self.term_chunk_table.setRowCount(len(rows))
        for r, (chunk, score) in enumerate(rows):
            panels = len(self.chunk_panel_sets.get(chunk, set()) & self.term_panel_sets.get(term, set()))
            self.term_chunk_table.setItem(r, 0, QTableWidgetItem(chunk))
            self.term_chunk_table.setItem(r, 1, QTableWidgetItem(self._kind_for_chunk(chunk)))
            self.term_chunk_table.setItem(r, 2, QTableWidgetItem(f"{score:.4f}"))
            self.term_chunk_table.setItem(r, 3, QTableWidgetItem(str(panels)))

    # ---------- Matching & hypotheses ----------
    def suggest_matching(self) -> None:
        G = nx.Graph()
        for c in self.filtered_candidates:
            chunk = c.display
            for term, wt in self.chunk_term_weights.get(chunk, {}).items():
                if wt <= 0:
                    continue
                G.add_edge(f"C::{chunk}", f"T::{term}", weight=wt)
        if G.number_of_edges() == 0:
            self.matching_suggestions = []
            self.refresh_hypothesis_tables()
            return
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
        sug: List[Tuple[str, str, float]] = []
        for a, b in matching:
            if a.startswith("C::"):
                ch, tm = a[3:], b[3:]
            else:
                ch, tm = b[3:], a[3:]
            wt = G[a][b].get("weight", 0.0)
            sug.append((ch, tm, wt))
        sug.sort(key=lambda x: (-x[2], x[0], x[1]))
        self.matching_suggestions = sug
        self.refresh_hypothesis_tables()
        self.summary_box.append(f"Suggested one-to-one matching with {len(sug)} links.")

    def add_hypothesis_from_graph_choice(self) -> None:
        chunk = (self.current_hyp_chunk or "").strip()
        if not chunk:
            QMessageBox.information(self, "Choose a chunk", "Click a chunk node first.")
            return
        term = self.hyp_term_combo.currentData()
        if not term:
            QMessageBox.information(self, "Choose a term", "Pick one of the candidate terms for this chunk.")
            return
        self.current_assignment[chunk] = term
        candidate = next((c for c in self.filtered_candidates if c.display == chunk), None)
        if not candidate:
            return
        score = self.chunk_term_weights.get(chunk, {}).get(term, 0.0)
        gloss = self.custom_gloss_edit.text().strip() or term
        source = "graph-manual" if gloss == term else "graph-manual-custom"
        self.working_lexicon[chunk] = HypothesisRecord(
            chunk_display=chunk,
            chunk_ids=candidate.ids,
            proposed_gloss=gloss,
            evidence_score=score,
            source=source,
            notes=f"selected_term={term}",
        )
        self.refresh_hypothesis_tables()
        self.draw_hypothesis_graph()

    def on_hyp_graph_click(self, event) -> None:
        if event.inaxes is None or not self.hyp_graph_positions:
            return
        best = None
        best_d2 = 1e9
        for node, (x, y) in self.hyp_graph_positions.items():
            if not node.startswith("C::"):
                continue
            d2 = (event.xdata - x) ** 2 + (event.ydata - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = node
        if best is None or best_d2 > 0.0035:
            return
        self.select_hyp_chunk(best[3:])

    def select_hyp_chunk(self, chunk: str) -> None:
        self.current_hyp_chunk = chunk
        self.hyp_selected_chunk_label.setText(chunk)
        opts = self.chunk_choice_options.get(chunk, [])
        self._updating_hyp_combo = True
        self.hyp_term_combo.clear()
        current_term = self.current_assignment.get(chunk, "")
        for idx, (term, wt) in enumerate(opts):
            label = f"#{idx+1}  {term}  ({wt:.4f})"
            self.hyp_term_combo.addItem(label, term)
            if term == current_term:
                self.hyp_term_combo.setCurrentIndex(idx)
        self._updating_hyp_combo = False
        rec = self.working_lexicon.get(chunk)
        self.custom_gloss_edit.setText(rec.proposed_gloss if rec else "")
        self.update_hyp_graph_detail(chunk)
        # sync selection in table
        for r in range(self.hyp_assignment_table.rowCount()):
            item = self.hyp_assignment_table.item(r, 0)
            if item and item.text() == chunk:
                self.hyp_assignment_table.selectRow(r)
                break
        self.draw_hypothesis_graph()

    def hyp_term_changed(self, index: int) -> None:
        if self._updating_hyp_combo or not self.current_hyp_chunk:
            return
        term = self.hyp_term_combo.currentData()
        if not term:
            return
        self.current_assignment[self.current_hyp_chunk] = term
        candidate = next((c for c in self.filtered_candidates if c.display == self.current_hyp_chunk), None)
        if candidate:
            score = self.chunk_term_weights.get(self.current_hyp_chunk, {}).get(term, 0.0)
            gloss = self.working_lexicon.get(self.current_hyp_chunk, HypothesisRecord(self.current_hyp_chunk, candidate.ids, term, score)).proposed_gloss
            if not gloss or gloss == self.current_hyp_chunk or gloss in self.chunk_term_weights.get(self.current_hyp_chunk, {}):
                gloss = term
            self.working_lexicon[self.current_hyp_chunk] = HypothesisRecord(
                chunk_display=self.current_hyp_chunk,
                chunk_ids=candidate.ids,
                proposed_gloss=gloss,
                evidence_score=score,
                source="graph-manual",
                notes=f"selected_term={term}",
            )
        self.refresh_hypothesis_tables()
        self.update_hyp_graph_detail(self.current_hyp_chunk)
        self.draw_hypothesis_graph()

    def update_hyp_graph_detail(self, chunk: str) -> None:
        cand = next((c for c in self.filtered_candidates if c.display == chunk), None)
        if not cand:
            self.hyp_graph_detail.clear(); return
        opts = self.chunk_choice_options.get(chunk, [])
        chosen = self.current_assignment.get(chunk, "")
        rank = next((i + 1 for i, (t, _) in enumerate(opts) if t == chosen), None)
        lines = [
            f"Chunk: {cand.display}",
            f"Glyph IDs: {' '.join(cand.ids)}",
            f"Kind: {cand.kind}",
            f"Count: {cand.count}",
            f"Chosen term: {chosen or '(none)'}",
            f"Chosen rank: {rank if rank is not None else '(none)'}",
            "",
            "Top candidate terms:",
        ]
        for i, (term, wt) in enumerate(opts, start=1):
            prefix = "*" if term == chosen else "-"
            lines.append(f"{prefix} #{i} {term}: {wt:.4f}")
        self.hyp_graph_detail.setPlainText("\n".join(lines))

    def _edge_color_for_choice(self, chunk: str, term: str) -> str:
        opts = self.chunk_choice_options.get(chunk, [])
        if not opts:
            return "#444444"
        top_term = opts[0][0]
        if term == top_term:
            return "#000000"
        weights = [wt for _, wt in opts]
        w = dict(opts).get(term, 0.0)
        lo, hi = min(weights), max(weights)
        if hi <= lo:
            return "#7c3aed"
        frac = (w - lo) / (hi - lo)
        # red -> amber -> green for weaker to stronger alternates
        if frac < 0.5:
            r, g = 220, int(80 + 200 * frac)
        else:
            r, g = int(220 - 200 * (frac - 0.5) * 2), 180
        return f"#{r:02x}{g:02x}55"

    def cleanup_hypothesis_layout(self) -> None:
        self.hyp_clean_layout = True
        self.draw_hypothesis_graph()

    def _compute_clean_term_positions(self, chunks: List[str], chosen_terms: List[str]) -> Dict[str, float]:
        """
        Layout heuristic for the right-side concept nodes.

        - If a term has one incoming chunk, place it directly across from that chunk.
        - If a term has multiple incoming chunks, place it at the average y of those chunks.
        - Then nudge terms to avoid label collisions while preserving order.
        """
        if not chosen_terms:
            return {}

        if len(chunks) == 1:
            chunk_y = {chunks[0]: 0.5}
        else:
            gap = 0.90 / max(1, len(chunks) - 1)
            chunk_y = {ch: 0.95 - i * gap for i, ch in enumerate(chunks)}

        targets: List[Tuple[str, float]] = []
        for term in chosen_terms:
            attached = [chunk_y[ch] for ch in chunks if self.current_assignment.get(ch) == term]
            if attached:
                target_y = sum(attached) / len(attached)
            else:
                target_y = 0.5
            targets.append((term, target_y))

        # Preserve top-to-bottom order by desired y
        targets.sort(key=lambda x: (-x[1], x[0]))

        # Forward pass: enforce a minimum gap
        min_gap = 0.035
        placed: Dict[str, float] = {}
        prev_y = 1.02
        for term, target_y in targets:
            y = min(target_y, prev_y - min_gap)
            placed[term] = y
            prev_y = y

        # Reverse pass: keep nodes from collapsing near the bottom
        prev_y = -0.02
        for term, _ in reversed(targets):
            y = max(placed[term], prev_y + min_gap)
            placed[term] = y
            prev_y = y

        # Clamp to drawing window
        for term in placed:
            placed[term] = min(0.95, max(0.05, placed[term]))

        return placed

    def draw_hypothesis_graph(self) -> None:
        self.hyp_graph_canvas.fig.clear()
        ax = self.hyp_graph_canvas.fig.add_subplot(111)
        ax.set_axis_off()
        self.hyp_graph_positions = {}
        G = nx.Graph()
        chunks = [c.display for c in self.filtered_candidates]
        chunks = [c.display for c in self.filtered_candidates]

        chosen_terms = []
        for ch in chunks:
            term = self.current_assignment.get(ch)
            if term and term not in chosen_terms:
                chosen_terms.append(term)

        # Left side: keep chunk order stable
        if len(chunks) == 1:
            chunk_gap = 0.0
        else:
            chunk_gap = 0.90 / max(1, len(chunks) - 1)

        for i, ch in enumerate(chunks):
            y = 0.95 - i * chunk_gap
            node = f"C::{ch}"
            G.add_node(node, ntype="chunk", label=ch)
            self.hyp_graph_positions[node] = (0.12, y)

        # Right side: either alphabetical (old) or cleaned layout
        if self.hyp_clean_layout:
            term_y = self._compute_clean_term_positions(chunks, chosen_terms)
            chosen_terms = sorted(chosen_terms, key=lambda t: -term_y.get(t, 0.5))
        else:
            chosen_terms = sorted(chosen_terms)
            term_y = {}
            if len(chosen_terms) == 1:
                term_y[chosen_terms[0]] = 0.5
            else:
                term_gap = 0.90 / max(1, len(chosen_terms) - 1)
                for i, term in enumerate(chosen_terms):
                    term_y[term] = 0.95 - i * term_gap

        for term in chosen_terms:
            y = term_y.get(term, 0.5)
            node = f"T::{term}"
            G.add_node(node, ntype="term", label=term)
            self.hyp_graph_positions[node] = (0.88, y)
        for ch in chunks:
            term = self.current_assignment.get(ch)
            if not term:
                continue
            wt = self.chunk_term_weights.get(ch, {}).get(term, 0.0)
            G.add_edge(f"C::{ch}", f"T::{term}", weight=wt, color=self._edge_color_for_choice(ch, term))
        # draw
        widths = [1.5 + 2.2 * max(0.0, G.edges[e].get("weight", 0.0)) for e in G.edges()]
        colors = [G.edges[e].get("color", "#444444") for e in G.edges()]
        nx.draw_networkx_edges(G, self.hyp_graph_positions, ax=ax, width=widths, edge_color=colors, alpha=0.9)
        node_colors = []
        for n in G.nodes():
            if n.startswith("C::"):
                node_colors.append("#f59e0b" if n[3:] == self.current_hyp_chunk else "#2563eb")
            else:
                node_colors.append("#059669")
        nx.draw_networkx_nodes(G, self.hyp_graph_positions, ax=ax, nodelist=list(G.nodes()), node_size=900, node_color=node_colors, alpha=0.96)
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, self.hyp_graph_positions, ax=ax, labels=labels, font_size=8, font_color="white")
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
        self.hyp_graph_canvas.draw_idle()

    def hyp_assignment_row_changed(self, currentRow: int, currentColumn: int, previousRow: int, previousColumn: int) -> None:
        item = self.hyp_assignment_table.item(currentRow, 0)
        if not item:
            return
        self.select_hyp_chunk(item.text())

    def reset_hypotheses_to_top_edges(self) -> None:
        self._initialize_hypothesis_state()
        self.hyp_clean_layout = False
        self.refresh_hypothesis_tables()
        self.draw_hypothesis_graph()

    def adopt_matching_suggestions(self) -> None:
        for chunk, term, wt in self.matching_suggestions:
            candidate = next((c for c in self.filtered_candidates if c.display == chunk), None)
            if not candidate:
                continue
            self.current_assignment[chunk] = term
            self.working_lexicon[chunk] = HypothesisRecord(
                chunk_display=chunk,
                chunk_ids=candidate.ids,
                proposed_gloss=term,
                evidence_score=wt,
                source="matching",
                notes=f"selected_term={term}",
            )
        self.refresh_hypothesis_tables()
        self.draw_hypothesis_graph()

    def remove_selected_hypothesis(self) -> None:
        row = self.hyp_assignment_table.currentRow()
        if row < 0:
            return
        item = self.hyp_assignment_table.item(row, 0)
        if not item:
            return
        chunk = item.text()
        self.working_lexicon.pop(chunk, None)
        self.current_assignment.pop(chunk, None)
        self.refresh_hypothesis_tables()
        self.draw_hypothesis_graph()

    def clear_hypotheses(self) -> None:
        self.working_lexicon.clear()
        self.current_assignment.clear()
        self.refresh_hypothesis_tables()
        self.draw_hypothesis_graph()

    def refresh_hypothesis_tables(self) -> None:
        rows = []
        for c in self.filtered_candidates:
            chunk = c.display
            term = self.current_assignment.get(chunk, "")
            wt = self.chunk_term_weights.get(chunk, {}).get(term, 0.0) if term else 0.0
            opts = self.chunk_choice_options.get(chunk, [])
            rank = next((i + 1 for i, (t, _) in enumerate(opts) if t == term), None)
            gloss = self.working_lexicon.get(chunk).proposed_gloss if chunk in self.working_lexicon else term
            rows.append((chunk, term, wt, rank or 0, gloss))
        rows.sort(key=lambda x: (x[3] if x[3] else 999, -x[2], x[0]))
        self.hyp_assignment_table.setRowCount(len(rows))
        for r, (chunk, term, wt, rank, gloss) in enumerate(rows):
            self.hyp_assignment_table.setItem(r, 0, QTableWidgetItem(chunk))
            self.hyp_assignment_table.setItem(r, 1, QTableWidgetItem(term))
            self.hyp_assignment_table.setItem(r, 2, QTableWidgetItem(f"{wt:.4f}"))
            self.hyp_assignment_table.setItem(r, 3, QTableWidgetItem(str(rank) if rank else ""))
            self.hyp_assignment_table.setItem(r, 4, QTableWidgetItem(gloss))

        self.match_table.setRowCount(len(self.matching_suggestions))
        for r, (chunk, term, wt) in enumerate(self.matching_suggestions):
            self.match_table.setItem(r, 0, QTableWidgetItem(chunk))
            self.match_table.setItem(r, 1, QTableWidgetItem(term))
            self.match_table.setItem(r, 2, QTableWidgetItem(f"{wt:.4f}"))

        total = 0.0
        glosses = []
        for chunk, term in self.current_assignment.items():
            total += self.chunk_term_weights.get(chunk, {}).get(term, 0.0)
            gloss = self.working_lexicon.get(chunk).proposed_gloss if chunk in self.working_lexicon else term
            if gloss:
                glosses.append(gloss)
        gloss_counts = Counter(glosses)
        duplicate_penalty = sum(v - 1 for v in gloss_counts.values() if v > 1) * 0.5
        coverage = len(self.current_assignment) / max(1, len(self.filtered_candidates))
        final_score = total + 2.0 * coverage - duplicate_penalty
        self.score_label.setText(
            f"Assignments: {len(self.current_assignment)} | coverage: {coverage:.2%} | evidence sum: {total:.3f} | duplicate penalty: {duplicate_penalty:.3f} | overall score: {final_score:.3f}"
        )
        self.draw_hypothesis_graph()

    # ---------- Saving ----------
    def save_working_lexicon(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save working lexicon JSON", "working_lexicon.json", "JSON files (*.json)")
        if not path:
            return
        payload = {
            "dataset": os.path.basename(self.dataset_path or ""),
            "inventory": os.path.basename(self.inventory_path or ""),
            "entries": [
                {
                    "chunk_display": rec.chunk_display,
                    "chunk_ids": rec.chunk_ids,
                    "selected_term": self.current_assignment.get(rec.chunk_display, ""),
                    "proposed_gloss": rec.proposed_gloss,
                    "evidence_score": rec.evidence_score,
                    "source": rec.source,
                    "notes": rec.notes,
                }
                for rec in sorted(self.working_lexicon.values(), key=lambda x: x.chunk_display)
            ],
            "matching_suggestions": [
                {"chunk_display": ch, "suggested_gloss": tm, "weight": wt}
                for ch, tm, wt in self.matching_suggestions
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.summary_box.append(f"Saved working lexicon to {path}")

    def export_association_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export association report JSON", "chunk_association_report.json", "JSON files (*.json)")
        if not path:
            return
        rows = []
        for c in self.filtered_candidates:
            rows.append({
                "chunk_display": c.display,
                "chunk_ids": c.ids,
                "kind": c.kind,
                "count": c.count,
                "score": c.score,
                "top_terms": [
                    {"term": term, "association": wt}
                    for term, wt in sorted(self.chunk_term_weights.get(c.display, {}).items(), key=lambda x: (-x[1], x[0]))[:20]
                ],
                "panels": sorted(self.chunk_panel_sets.get(c.display, set())),
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f, indent=2)
        self.summary_box.append(f"Exported association report to {path}")

    def select_random_sentence(self) -> None:
        if not self.sentences:
            QMessageBox.information(self, "Load data", "Load the master dataset first.")
            return
        self.challenge_current_sentence = random.choice(self.sentences)
        sn = self.challenge_current_sentence.get("sentence_number", "?")
        self.challenge_sentence_num_label.setText(f"Sentence: {sn}")
        self.challenge_glyph_label.setText(segmented_symbol_display(self.challenge_current_sentence))
        self.challenge_student_interp.clear()
        self.challenge_reveal_text.clear()
        self.challenge_reveal_btn.setEnabled(True)

    def reveal_true_sentence(self) -> None:
        if not self.challenge_current_sentence:
            QMessageBox.information(self, "No sentence selected", "Choose a random sentence first.")
            return
        s = self.challenge_current_sentence
        sn = s.get("sentence_number", "?")
        english = s.get("instructor_view", {}).get("english_translation", "(no translation found)")
        sv = s.get("student_view", {})
        panel_id = sv.get("linked_panel_id")
        panel_caption = sv.get("linked_panel_caption")
        loose = sv.get("loose_find", False)
        loose_note = sv.get("loose_find_note")

        lines = [f"Sentence {sn}", "", f"Original English sentence:", english]
        if panel_id and panel_caption:
            lines.extend(["", f"Linked panel: {panel_id}", panel_caption])
        elif loose and loose_note:
            lines.extend(["", "Loose-find note:", loose_note])
        self.challenge_reveal_text.setPlainText("\n".join(lines))

    # ---------- Helpers ----------
    def _kind_for_chunk(self, display: str) -> str:
        cand = next((c for c in self.filtered_candidates if c.display == display), None)
        return cand.kind if cand else "candidate"


def main() -> None:
    app = QApplication(sys.argv)
    win = Stage3GraphWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
