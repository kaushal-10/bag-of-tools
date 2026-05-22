"""
ONNX Model Visualizer & Splitter
---------------------------------
Features:
  • Load & validate ONNX models
  • Interactive graph visualization (zoom / pan / click)
  • Node search and filtering by op-type
  • Node / tensor / model-metadata inspector
  • Split model at any node boundary and save both parts
  • Export graph as PNG
  • Dark / light theme toggle
"""

import sys
import os
import re
import copy
import collections
import colorsys
from pathlib import Path

import numpy as np
import networkx as nx

try:
    import onnx
    import onnx.helper as oh
    import onnx.checker
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QTabWidget, QTextEdit,
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QScrollArea, QFrame, QToolBar, QAction, QStatusBar, QProgressBar,
    QCheckBox, QSizePolicy, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRectF, QPointF
from PyQt5.QtGui import (
    QFont, QIcon, QPalette, QColor, QPixmap,
    QPainter, QPen, QBrush, QLinearGradient, QPolygonF,
    QPainterPath, QPainterPathStroker, QTransform, QImage,
)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPathItem
)


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette helpers
# ──────────────────────────────────────────────────────────────────────────────

OP_COLORS = {
    "Conv":       "#4A90D9",
    "Relu":       "#7ED321",
    "MaxPool":    "#9B59B6",
    "AveragePool":"#8E44AD",
    "BatchNormalization": "#E67E22",
    "Gemm":       "#E74C3C",
    "MatMul":     "#C0392B",
    "Add":        "#27AE60",
    "Mul":        "#16A085",
    "Reshape":    "#2980B9",
    "Transpose":  "#1ABC9C",
    "Concat":     "#F39C12",
    "Flatten":    "#D35400",
    "Softmax":    "#8E44AD",
    "Sigmoid":    "#2ECC71",
    "Dropout":    "#95A5A6",
    "LSTM":       "#C0392B",
    "GRU":        "#922B21",
    "Pad":        "#1A5276",
    "Slice":      "#0E6655",
    "Gather":     "#784212",
    "Unsqueeze":  "#4D5656",
    "Squeeze":    "#4D5656",
    "Split":      "#A04000",
    "Resize":     "#154360",
    "Upsample":   "#154360",
}
DEFAULT_OP_COLOR = "#607D8B"

DARK_STYLE = """
QMainWindow, QDialog { background-color: #1e1e1e; color: #d4d4d4; }
QWidget { background-color: #1e1e1e; color: #d4d4d4; }
QTreeWidget { background-color: #252526; color: #d4d4d4; border: 1px solid #3c3c3c; }
QTreeWidget::item:selected { background-color: #094771; }
QTabWidget::pane { border: 1px solid #3c3c3c; }
QTabBar::tab { background: #2d2d2d; color: #abb2bf; padding: 6px 14px; }
QTabBar::tab:selected { background: #094771; color: #ffffff; }
QLineEdit { background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555; padding: 4px; border-radius: 3px; }
QPushButton { background-color: #0e639c; color: white; border: none; padding: 6px 14px; border-radius: 4px; }
QPushButton:hover { background-color: #1177bb; }
QPushButton:pressed { background-color: #094771; }
QPushButton:disabled { background-color: #3c3c3c; color: #666; }
QTextEdit { background-color: #252526; color: #d4d4d4; border: 1px solid #3c3c3c; font-family: monospace; }
QGroupBox { border: 1px solid #3c3c3c; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { color: #9cdcfe; }
QComboBox { background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555; padding: 4px; }
QLabel { color: #d4d4d4; }
QScrollBar:vertical { background: #252526; width: 12px; }
QScrollBar::handle:vertical { background: #555; border-radius: 6px; }
QStatusBar { background-color: #007acc; color: white; }
QToolBar { background-color: #333333; border: none; padding: 2px; }
"""

LIGHT_STYLE = """
QStatusBar { background-color: #007acc; color: white; }
QTreeWidget::item:selected { background-color: #cce8ff; }
QPushButton { background-color: #0078d4; color: white; border: none; padding: 6px 14px; border-radius: 4px; }
QPushButton:hover { background-color: #1084d8; }
QPushButton:disabled { background-color: #ccc; color: #888; }
QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# ONNX helpers
# ──────────────────────────────────────────────────────────────────────────────

def dtype_name(elem_type: int) -> str:
    mapping = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32",  7: "int64",  8: "string", 9: "bool",
    10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
    14: "complex64", 15: "complex128", 16: "bfloat16",
    }
    return mapping.get(elem_type, f"type_{elem_type}")


def tensor_shape_str(type_proto) -> str:
    try:
        shape = type_proto.tensor_type.shape
        dims = []
        for d in shape.dim:
            dims.append(str(d.dim_value) if d.dim_value > 0 else (d.dim_param or "?"))
        return f"[{', '.join(dims)}]"
    except Exception:
        return "unknown"


def node_name(node, idx: int) -> str:
    """Return a stable, unique name for an ONNX node.
    Uses the node's own name when set, otherwise a deterministic index-based id.
    This must be used consistently everywhere instead of id(node)."""
    return node.name if node.name else f"{node.op_type}_{idx}"


def build_graph(model) -> nx.DiGraph:
    """Build a directed graph from an ONNX model."""
    G = nx.DiGraph()
    for idx, node in enumerate(model.graph.node):
        name = node_name(node, idx)
        G.add_node(name, op_type=node.op_type,
                   inputs=list(node.input),
                   outputs=list(node.output),
                   node_ref=node,
                   node_idx=idx)

    output_map = {}  # tensor_name -> node_name that produces it
    for idx, node in enumerate(model.graph.node):
        name = node_name(node, idx)
        for out in node.output:
            output_map[out] = name

    for idx, node in enumerate(model.graph.node):
        dst = node_name(node, idx)
        for inp in node.input:
            if inp in output_map:
                src = output_map[inp]
                G.add_edge(src, dst, tensor=inp)
    return G


# Layout constants (pixels)
NODE_W  = 160
NODE_H  = 52
X_GAP   = 220   # centre-to-centre horizontal
Y_GAP   = 110   # centre-to-centre vertical


def hierarchical_layout(G: nx.DiGraph) -> dict:
    """Topological layered layout, top-to-bottom, centred per layer. Returns pixel coords."""
    if len(G.nodes) == 0:
        return {}
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        order = list(G.nodes)

    layer: dict = {}
    for n in order:
        preds = list(G.predecessors(n))
        layer[n] = max((layer[p] for p in preds), default=-1) + 1

    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for n, l in layer.items():
        buckets[l].append(n)

    pos = {}
    for l, nodes in buckets.items():
        total_w = len(nodes) * X_GAP
        start_x = -total_w / 2 + X_GAP / 2
        for i, n in enumerate(nodes):
            pos[n] = (start_x + i * X_GAP, l * Y_GAP)
    return pos


def split_model_at_node(model, split_node_name: str):
    """
    Split an ONNX model into two parts at a given node.

    Part 1 : original inputs  →  ALL outputs of split_node (and ancestors)
    Part 2 : ALL part-1 cross-boundary tensors as inputs  →  original outputs
    """
    # Build stable name map  idx → stable_name
    name_map = {idx: node_name(n, idx) for idx, n in enumerate(model.graph.node)}

    # Find the split node by stable name
    split_idx = None
    for idx, n in enumerate(model.graph.node):
        if name_map[idx] == split_node_name:
            split_idx = idx
            break
    if split_idx is None:
        raise ValueError(f"Node '{split_node_name}' not found in model.")

    split_node = model.graph.node[split_idx]

    # Build traversal graph using stable names
    G = build_graph(model)

    # Collect all value_info for tensor lookups
    all_value_info = {vi.name: vi for vi in model.graph.value_info}
    all_value_info.update({vi.name: vi for vi in model.graph.input})
    all_value_info.update({vi.name: vi for vi in model.graph.output})

    # ---------- PART 1 ----------
    # All nodes up to and including the split node
    part1_node_names = set(nx.ancestors(G, split_node_name)) | {split_node_name}
    part1_nodes = [n for idx, n in enumerate(model.graph.node)
                   if name_map[idx] in part1_node_names]

    # Collect ALL tensors produced by part-1 nodes
    part1_produced = set()
    for n in part1_nodes:
        part1_produced.update(out for out in n.output if out)

    # Part-1 outputs: every tensor produced by part-1 that crosses into part-2
    # (or, if nothing crosses, just the split node's outputs)
    # Initializers used in part 1
    part1_tensors_needed = set()
    for n in part1_nodes:
        part1_tensors_needed.update(inp for inp in n.input if inp)
    p1_initializers = [ini for ini in model.graph.initializer
                       if ini.name in part1_tensors_needed]

    # Part-1 outputs are the split node's non-empty outputs
    split_outputs = [t for t in split_node.output if t]

    def _make_vi(name):
        if name in all_value_info:
            return all_value_info[name]
        return oh.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)

    p1_output_vis = [_make_vi(t) for t in split_outputs]

    p1_graph = oh.make_graph(
        nodes=part1_nodes,
        name="part1",
        inputs=list(model.graph.input),
        outputs=p1_output_vis,
        initializer=p1_initializers,
    )
    p1_model = oh.make_model(p1_graph, opset_imports=model.opset_import)
    p1_model.ir_version = model.ir_version

    # ---------- PART 2 ----------
    # nx.descendants never includes the node itself
    part2_node_names = set(nx.descendants(G, split_node_name))
    part2_nodes = [n for idx, n in enumerate(model.graph.node)
                   if name_map[idx] in part2_node_names]

    # Determine ALL cross-boundary inputs to part2:
    # any tensor consumed by part2 that is produced by part1 (not an initializer).
    init_names = {ini.name for ini in model.graph.initializer}
    part2_consumed = set()
    for n in part2_nodes:
        part2_consumed.update(inp for inp in n.input if inp)

    boundary_tensors = part2_consumed & part1_produced  # cross-edge tensors

    # Also include original model inputs consumed directly by part2 nodes
    # (rare branching case where a model input bypasses the split node)
    model_input_names = {vi.name for vi in model.graph.input}
    for t in part2_consumed:
        if t in model_input_names and t not in part1_produced:
            boundary_tensors.add(t)

    p2_input_vis = [_make_vi(t) for t in sorted(boundary_tensors)]

    part2_tensors_needed = set(inp for n in part2_nodes
                               for inp in n.input if inp)
    p2_initializers = [ini for ini in model.graph.initializer
                       if ini.name in part2_tensors_needed]

    p2_graph = oh.make_graph(
        nodes=part2_nodes,
        name="part2",
        inputs=p2_input_vis,
        outputs=list(model.graph.output),
        initializer=p2_initializers,
    )
    p2_model = oh.make_model(p2_graph, opset_imports=model.opset_import)
    p2_model.ir_version = model.ir_version

    return p1_model, p2_model


def split_model_at_tensors(model, cut_tensors: list):
    """
    Split an ONNX model by cutting a specific set of tensor edges.

    cut_tensors : list of tensor names (edges) to sever.

    The algorithm:
        Part 1 = every node that *produces* a cut tensor, plus all their ancestors.
                 Outputs  = the cut tensors themselves.
        Part 2 = every remaining node (all descendants of the cut boundary).
                 Inputs   = the cut tensors + any model-level inputs consumed
                            directly by Part-2 nodes (bypass branches).

    This lets you "cut the net" at exactly the connections you choose.
    """
    if not cut_tensors:
        raise ValueError("No cut tensors specified.")

    cut_set = set(cut_tensors)

    G = build_graph(model)
    name_map   = {idx: node_name(n, idx) for idx, n in enumerate(model.graph.node)}
    # tensor → stable node name that produces it
    output_to_node: dict = {}
    for idx, n in enumerate(model.graph.node):
        nm = name_map[idx]
        for out in n.output:
            if out:
                output_to_node[out] = nm

    # ── Part 1: producers of cut tensors + all their ancestors ───────────────
    part1_names: set = set()
    for t in cut_set:
        if t in output_to_node:
            producer = output_to_node[t]
            part1_names.add(producer)
            part1_names.update(nx.ancestors(G, producer))

    # ── Part 2: every node NOT in part 1 ────────────────────────────────────
    part2_names = set(name_map.values()) - part1_names

    # Preserve original graph ordering for serialisation
    part1_nodes = [n for idx, n in enumerate(model.graph.node)
                   if name_map[idx] in part1_names]
    part2_nodes = [n for idx, n in enumerate(model.graph.node)
                   if name_map[idx] in part2_names]

    all_value_info: dict = {vi.name: vi for vi in model.graph.value_info}
    all_value_info.update({vi.name: vi for vi in model.graph.input})
    all_value_info.update({vi.name: vi for vi in model.graph.output})

    def _make_vi(name):
        if name in all_value_info:
            return all_value_info[name]
        return oh.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)

    # ── Compute ALL cross-boundary tensors ───────────────────────────────────
    # Any tensor produced by Part 1 and consumed by Part 2 (excluding initializers)
    # must be an output of Part 1 AND an input of Part 2.  Using only the explicit
    # cut_tensors misses ancestor-node outputs that also feed into Part 2 nodes.
    init_names = {ini.name for ini in model.graph.initializer}
    model_input_names = {vi.name for vi in model.graph.input}

    part1_produced: set = set()
    for n in part1_nodes:
        part1_produced.update(out for out in n.output if out)

    p2_consumed: set = set(inp for n in part2_nodes for inp in n.input if inp)

    # All tensors that cross Part1 → Part2 (not initializers, not raw model inputs)
    cross_boundary = (part1_produced & p2_consumed) - init_names

    # Order: explicit cut tensors first (user's order), then remaining cross-boundary
    p1_output_names: list = []
    seen_out: set = set()
    for t in cut_tensors:
        if t in cross_boundary and t not in seen_out:
            p1_output_names.append(t)
            seen_out.add(t)
    for t in sorted(cross_boundary - seen_out):
        p1_output_names.append(t)
        seen_out.add(t)
    # Also expose any original model outputs produced inside Part 1
    for vi in model.graph.output:
        if vi.name in part1_produced and vi.name not in seen_out:
            p1_output_names.append(vi.name)
            seen_out.add(vi.name)

    p1_output_vis = [_make_vi(t) for t in p1_output_names]

    # ── Build Part 1 ────────────────────────────────────────────────────────
    p1_init_names = set(inp for n in part1_nodes for inp in n.input if inp)
    p1_initializers = [ini for ini in model.graph.initializer
                       if ini.name in p1_init_names]

    p1_graph = oh.make_graph(
        nodes=part1_nodes,
        name="part1",
        inputs=list(model.graph.input),
        outputs=p1_output_vis,
        initializer=p1_initializers,
    )
    p1_model = oh.make_model(p1_graph, opset_imports=model.opset_import)
    p1_model.ir_version = model.ir_version

    # ── Build Part 2 ────────────────────────────────────────────────────────
    # Part 2 inputs = ALL cross-boundary tensors (same set/order as p1 outputs)
    # + any original model inputs consumed directly by Part 2 (bypass branches)
    boundary_inputs = []
    seen_bi: set = set()
    for t in p1_output_names:
        if t in p2_consumed and t not in seen_bi:
            boundary_inputs.append(_make_vi(t))
            seen_bi.add(t)
    # original model inputs that Part 2 needs directly
    for t in sorted(p2_consumed):
        if t in model_input_names and t not in seen_bi and t not in init_names:
            boundary_inputs.append(_make_vi(t))
            seen_bi.add(t)

    p2_init_names = set(inp for n in part2_nodes for inp in n.input if inp)
    p2_initializers = [ini for ini in model.graph.initializer
                       if ini.name in p2_init_names]

    p2_graph = oh.make_graph(
        nodes=part2_nodes,
        name="part2",
        inputs=boundary_inputs,
        outputs=list(model.graph.output),
        initializer=p2_initializers,
    )
    p2_model = oh.make_model(p2_graph, opset_imports=model.opset_import)
    p2_model.ir_version = model.ir_version

    return p1_model, p2_model


# ──────────────────────────────────────────────────────────────────────────────
# Model-loading worker thread
# ──────────────────────────────────────────────────────────────────────────────

class LoadWorker(QThread):
    finished = pyqtSignal(object, str)   # model, error_msg
    progress = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.progress.emit("Loading model…")
            model = onnx.load(self.path)
            self.progress.emit("Checking model…")
            onnx.checker.check_model(model)
            self.finished.emit(model, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Qt-native graph canvas  (QGraphicsView → smooth zoom / pan)
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_qcolor(h: str) -> QColor:
    return QColor(h)


def _darker(hex_color: str, factor: float = 0.60) -> QColor:
    """Return a darker shade of a hex colour."""
    try:
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        dr, dg, db = colorsys.hsv_to_rgb(h, s, v * factor)
        return QColor(int(dr * 255), int(dg * 255), int(db * 255))
    except Exception:
        return _hex_to_qcolor(hex_color)


class NodeItem(QGraphicsItem):
    """A single operator node drawn as a rounded rect with a header band."""

    HEADER_RATIO = 0.42
    RADIUS = 7

    def __init__(self, name: str, op_type: str, base_color: str, on_moved=None):
        super().__init__()
        self.name = name
        self.op_type = op_type
        self.base_color = base_color
        self._selected_node = False
        self._highlight = None   # None | "part1" | "part2"
        self._on_moved = on_moved
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        return QRectF(-NODE_W / 2 - 2, -NODE_H / 2 - 2, NODE_W + 4, NODE_H + 4)

    def paint(self, painter: QPainter, option, widget=None):
        r = self.RADIUS
        rect = QRectF(-NODE_W / 2, -NODE_H / 2, NODE_W, NODE_H)
        header_h = NODE_H * self.HEADER_RATIO
        header_rect = QRectF(-NODE_W / 2, -NODE_H / 2, NODE_W, header_h)

        # resolve colours
        if self._highlight == "part1":
            face_hex = "#2563EB"
        elif self._highlight == "part2":
            face_hex = "#DC2626"
        else:
            face_hex = self.base_color
        face   = _hex_to_qcolor(face_hex)
        header = _darker(face_hex, 0.62)

        border_color = QColor("#FFD700") if self._selected_node else QColor(255, 255, 255, 55)
        border_lw    = 2.5 if self._selected_node else 1.0

        painter.setRenderHint(QPainter.Antialiasing)

        # body
        painter.setBrush(QBrush(face))
        painter.setPen(QPen(border_color, border_lw))
        painter.drawRoundedRect(rect, r, r)

        # header clip path
        path = QPainterPath()
        path.addRoundedRect(header_rect, r, r)
        painter.setClipPath(path)
        painter.setBrush(QBrush(header))
        painter.setPen(Qt.NoPen)
        # extend downward so bottom corners are square (clipped by body)
        painter.drawRect(QRectF(-NODE_W / 2, -NODE_H / 2, NODE_W, header_h + r))
        painter.setClipping(False)

        # op-type label
        painter.setPen(QColor("#ffffff"))
        f_header = QFont("Sans Serif", 8, QFont.Bold)
        painter.setFont(f_header)
        painter.drawText(
            QRectF(-NODE_W / 2, -NODE_H / 2, NODE_W, header_h),
            Qt.AlignCenter, self.op_type,
        )

        # node name label
        short = self.name if len(self.name) <= 22 else self.name[:20] + "…"
        painter.setPen(QColor(255, 255, 255, 180))
        f_body = QFont("Monospace", 6)
        painter.setFont(f_body)
        body_text_rect = QRectF(-NODE_W / 2 + 4, -NODE_H / 2 + header_h, NODE_W - 8,
                                NODE_H - header_h)
        painter.drawText(body_text_rect, Qt.AlignCenter, short)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and self._on_moved:
            self._on_moved(self.name)
        return super().itemChange(change, value)

    def set_selected(self, sel: bool):
        self._selected_node = sel
        self.update()

    def set_highlight(self, mode):
        self._highlight = mode
        self.update()


# ──────────────────────────────────────────────────────────────────────────────
# Edge item — a clickable bezier connection that can be "cut"
# ──────────────────────────────────────────────────────────────────────────────

class EdgeItem(QGraphicsPathItem):
    """Clickable bezier edge.  Click to toggle it as a cut boundary."""

    _COLOR_NORMAL  = QColor("#5a5a6a")
    _COLOR_DARK_BG = QColor("#5a5a6a")
    _COLOR_LITE_BG = QColor("#8888aa")
    _COLOR_HOVER   = QColor("#a0a0c0")
    _COLOR_CUT     = QColor("#FF8C00")   # orange when selected for cutting
    _COLOR_CUT_HOV = QColor("#FFB347")

    def __init__(self, bezier_path: QPainterPath, tensor_name: str,
                 dark: bool = True, on_toggle=None):
        super().__init__(bezier_path)
        self.tensor_name = tensor_name
        self._cut = False
        self._hovered = False
        self._on_toggle = on_toggle   # callable(tensor_name, is_cut)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1)            # draw below nodes
        self._dark = dark
        self._apply_pen()

    def _apply_pen(self):
        if self._cut:
            color = self._COLOR_CUT_HOV if self._hovered else self._COLOR_CUT
            pen = QPen(color, 2.5, Qt.DashLine)
        elif self._hovered:
            pen = QPen(self._COLOR_HOVER, 2.0)
        else:
            base = self._COLOR_DARK_BG if self._dark else self._COLOR_LITE_BG
            pen = QPen(base, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)

    def shape(self) -> QPainterPath:
        """Widen the hit area to 14 px so the line is easy to click."""
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(self.path())

    def set_cut(self, cut: bool):
        self._cut = cut
        self._apply_pen()
        self.update()

    def toggle_cut(self):
        self._cut = not self._cut
        self._apply_pen()
        self.update()
        if self._on_toggle:
            self._on_toggle(self.tensor_name, self._cut)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self._apply_pen()
        self.update()
        self.setCursor(Qt.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._apply_pen()
        self.update()
        self.unsetCursor()
        super().hoverLeaveEvent(event)


class GraphCanvas(QGraphicsView):
    nodeClicked   = pyqtSignal(str)
    edgeCutToggled = pyqtSignal(str, bool)   # tensor_name, is_cut

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._dark = True
        self._G = None
        self._pos = {}
        self._node_items: dict = {}   # name -> NodeItem
        self._edge_items: dict = {}   # tensor_name -> EdgeItem (last one if multi)
        self._edge_data: list = []    # (src, dst, tensor, EdgeItem, arrow_item, label_item)
        self._panning = False
        self._pan_start = None
        self._cut_tensors: set = set()
        self._selected_node = None
        self._highlighted_nodes: set = set()
        self._set_bg(True)

    def _set_bg(self, dark: bool):
        self._dark = dark
        self._scene.setBackgroundBrush(QBrush(QColor("#1e1e1e" if dark else "#f0f0f0")))

    # ── public API ────────────────────────────────────────────────────────────

    def draw_graph(self, G: nx.DiGraph, pos: dict, dark: bool = True,
                   preserve_cuts: bool = False):
        old_cuts = set(self._cut_tensors) if preserve_cuts else set()
        self._G = G
        self._pos = pos
        self._set_bg(dark)
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._edge_data.clear()
        self._cut_tensors.clear()

        if not G or not G.nodes:
            lbl = self._scene.addText("No model loaded")
            lbl.setDefaultTextColor(QColor("#d4d4d4"))
            return

        lbl_color  = QColor("#6a6a8a" if dark else "#888888")
        show_labels = len(G.edges) <= 150

        # ── edges ─────────────────────────────────────────────────────────────
        for src, dst, edata in G.edges(data=True):
            if src not in pos or dst not in pos:
                continue
            tensor = edata.get("tensor", "")
            sx, sy = pos[src]
            dx, dy = pos[dst]
            x0, y0 = sx, sy + NODE_H / 2
            x1, y1 = dx, dy - NODE_H / 2

            path = QPainterPath(QPointF(x0, y0))
            ctrl_dy = abs(y1 - y0) * 0.5
            path.cubicTo(x0, y0 + ctrl_dy, x1, y1 - ctrl_dy, x1, y1)

            edge_item = EdgeItem(path, tensor, dark=dark,
                                 on_toggle=self._on_edge_toggle)
            self._scene.addItem(edge_item)
            if tensor and tensor not in self._edge_items:
                self._edge_items[tensor] = edge_item

            # arrowhead (separate movable path item)
            arrow_item = self._make_arrow_item(x1, y1, 7,
                QColor("#5a5a6a" if dark else "#8888aa"))

            # tensor label
            label_item = None
            if show_labels and tensor:
                short = tensor[:20] + "…" if len(tensor) > 20 else tensor
                label_item = self._scene.addText(short, QFont("Monospace", 5))
                label_item.setDefaultTextColor(lbl_color)
                label_item.setPos((x0 + x1) / 2 + 4, (y0 + y1) / 2 - 8)
                label_item.setZValue(-1)

            self._edge_data.append((src, dst, tensor, edge_item, arrow_item, label_item))

        # ── nodes ─────────────────────────────────────────────────────────────
        for n, (x, y) in pos.items():
            op  = G.nodes[n].get("op_type", "?")
            col = OP_COLORS.get(op, DEFAULT_OP_COLOR)
            item = NodeItem(n, op, col, on_moved=self._node_moved)
            item.setPos(x, y)
            item.setZValue(1)
            self._scene.addItem(item)
            self._node_items[n] = item

        # restore cut state after redraw
        for t in old_cuts:
            self.set_cut_tensor(t, True)

        self._refresh_highlights()
        self.fitInView(self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40),
                       Qt.KeepAspectRatio)

    def _make_arrow_item(self, x: float, y: float, size: float,
                          color: QColor) -> QGraphicsPathItem:
        """Create and add a downward-pointing filled triangle arrowhead item."""
        poly = QPolygonF([
            QPointF(x,            y),
            QPointF(x - size / 2, y - size),
            QPointF(x + size / 2, y - size),
        ])
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        item = self._scene.addPath(path, QPen(Qt.NoPen), QBrush(color))
        item.setZValue(-1)
        return item

    def _on_edge_toggle(self, tensor: str, is_cut: bool):
        """Callback from EdgeItem when user clicks an edge."""
        if is_cut:
            self._cut_tensors.add(tensor)
        else:
            self._cut_tensors.discard(tensor)
        self.edgeCutToggled.emit(tensor, is_cut)

    def _node_moved(self, name: str):
        """Called by NodeItem when the user drags it; update pos dict and redraw edges."""
        if name in self._node_items:
            p = self._node_items[name].pos()
            self._pos[name] = (p.x(), p.y())
        for (src, dst, tensor, edge_item, arrow_item, label_item) in self._edge_data:
            if src == name or dst == name:
                self._update_edge_geometry(src, dst, edge_item, arrow_item, label_item)

    def _update_edge_geometry(self, src, dst, edge_item, arrow_item, label_item):
        """Recompute bezier path and arrowhead position from current node positions."""
        if src not in self._pos or dst not in self._pos:
            return
        sx, sy = self._pos[src]
        dx, dy = self._pos[dst]
        x0, y0 = sx, sy + NODE_H / 2
        x1, y1 = dx, dy - NODE_H / 2
        path = QPainterPath(QPointF(x0, y0))
        ctrl_dy = abs(y1 - y0) * 0.5
        path.cubicTo(x0, y0 + ctrl_dy, x1, y1 - ctrl_dy, x1, y1)
        edge_item.setPath(path)
        # update arrowhead
        size = 7
        poly = QPolygonF([
            QPointF(x1,          y1),
            QPointF(x1 - size/2, y1 - size),
            QPointF(x1 + size/2, y1 - size),
        ])
        ap = QPainterPath()
        ap.addPolygon(poly)
        ap.closeSubpath()
        arrow_item.setPath(ap)
        # update label
        if label_item:
            label_item.setPos((x0 + x1) / 2 + 4, (y0 + y1) / 2 - 8)

    def _refresh_highlights(self):
        for name, item in self._node_items.items():
            item.set_selected(name == self._selected_node)
            if self._highlighted_nodes:
                item.set_highlight("part1" if name in self._highlighted_nodes else "part2")
            else:
                item.set_highlight(None)

    def select_node(self, name: str):
        self._selected_node = name
        self._refresh_highlights()
        # scroll to node
        if name in self._node_items:
            self.ensureVisible(self._node_items[name], 80, 80)

    def highlight_split_preview(self, split_node: str = None,
                                cut_tensors: set = None):
        """Highlight part1 (blue) / part2 (red) based on either a cut-tensor set
        or a legacy single split-node."""
        if self._G is None:
            return
        if cut_tensors:
            # Build part1_names identically to split_model_at_tensors
            G = self._G
            # tensor → node that produces it
            output_to_node: dict = {}
            for n in G.nodes:
                for out in G.nodes[n].get("outputs", []):
                    if out:
                        output_to_node[out] = n
            part1: set = set()
            for t in cut_tensors:
                if t in output_to_node:
                    prod = output_to_node[t]
                    part1.add(prod)
                    part1.update(nx.ancestors(G, prod))
            self._highlighted_nodes = part1
        elif split_node:
            self._highlighted_nodes = (
                set(nx.ancestors(self._G, split_node)) | {split_node}
            )
        else:
            self._highlighted_nodes = set()
        self._refresh_highlights()

    def get_cut_tensors(self) -> set:
        return set(self._cut_tensors)

    def set_cut_tensor(self, tensor: str, cut: bool):
        if cut:
            self._cut_tensors.add(tensor)
        else:
            self._cut_tensors.discard(tensor)
        if tensor in self._edge_items:
            self._edge_items[tensor].set_cut(cut)

    def clear_cut_tensors(self):
        for t in list(self._cut_tensors):
            if t in self._edge_items:
                self._edge_items[t].set_cut(False)
        self._cut_tensors.clear()
        self._refresh_highlights()
        self.edgeCutToggled.emit("", False)   # signal "cleared"

    def clear_highlight(self):
        self._highlighted_nodes = set()
        self._selected_node = None
        self._refresh_highlights()

    def export_to_png(self, path: str):
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        img  = QImage(int(rect.width() * 2), int(rect.height() * 2), QImage.Format_ARGB32)
        img.fill(QColor("#1e1e1e" if self._dark else "#f0f0f0"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        self._scene.render(p, source=rect)
        p.end()
        img.save(path)

    # ── zoom / pan ────────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            for hit in self._scene.items(scene_pt):
                if isinstance(hit, EdgeItem):
                    hit.toggle_cut()
                    return
                if isinstance(hit, NodeItem):
                    self._selected_node = hit.name
                    self._refresh_highlights()
                    self.nodeClicked.emit(hit.name)
                    # let base class handle item drag
                    super().mousePressEvent(event)
                    return
            # empty space → start panning
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self._panning:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    # ── drag-and-drop pass-through (so MainWindow receives .onnx drops) ───────

    def dragEnterEvent(self, event):
        self.window().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        self.window().dragMoveEvent(event)

    def dropEvent(self, event):
        self.window().dropEvent(event)

    def keyPressEvent(self, event):
        """Ctrl+0 resets zoom; +/- zoom in/out."""
        if event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            self.fitInView(self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40),
                           Qt.KeepAspectRatio)
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.scale(1.2, 1.2)
        elif event.key() == Qt.Key_Minus:
            self.scale(1 / 1.2, 1 / 1.2)
        else:
            super().keyPressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Split dialog
# ──────────────────────────────────────────────────────────────────────────────

class SplitDialog(QDialog):
    def __init__(self, node_names: list, model_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split ONNX Model")
        self.setMinimumWidth(520)
        self.node_names = node_names
        self.model_stem = Path(model_path).stem if model_path else "model"
        self.chosen_node = None
        self.out_dir = str(Path(model_path).parent) if model_path else os.getcwd()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Node selector
        self.node_combo = QComboBox()
        self.node_combo.addItems(self.node_names)
        self.node_combo.setEditable(True)
        self.node_combo.currentTextChanged.connect(self._update_preview)
        form.addRow("Split after node:", self.node_combo)

        # Output directory
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(self.out_dir)
        self.dir_edit.textChanged.connect(self._update_preview)
        dir_btn = QPushButton("Browse…")
        dir_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(dir_btn)
        form.addRow("Output directory:", dir_row)

        layout.addLayout(form)

        # Preview
        preview_box = QGroupBox("Output file names (preview)")
        pv_layout = QVBoxLayout(preview_box)
        self.p1_label = QLabel()
        self.p2_label = QLabel()
        self.p1_label.setFont(QFont("monospace", 9))
        self.p2_label.setFont(QFont("monospace", 9))
        pv_layout.addWidget(self.p1_label)
        pv_layout.addWidget(self.p2_label)
        layout.addWidget(preview_box)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._update_preview()

    def _update_preview(self):
        node = self.node_combo.currentText()
        safe = re.sub(r"[^\w]", "_", node)[:32]
        d = self.dir_edit.text() or "."
        self.p1_label.setText(f"Part 1: {d}/{self.model_stem}__part1__{safe}.onnx")
        self.p2_label.setText(f"Part 2: {d}/{self.model_stem}__part2__{safe}.onnx")

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", self.out_dir)
        if d:
            self.dir_edit.setText(d)

    def get_params(self):
        node = self.node_combo.currentText()
        safe = re.sub(r"[^\w]", "_", node)[:32]
        d = self.dir_edit.text() or "."
        p1 = os.path.join(d, f"{self.model_stem}__part1__{safe}.onnx")
        p2 = os.path.join(d, f"{self.model_stem}__part2__{safe}.onnx")
        return node, p1, p2


# ──────────────────────────────────────────────────────────────────────────────
# Node info panel
# ──────────────────────────────────────────────────────────────────────────────

class NodeInfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("Select a node")
        self.title.setFont(QFont("Sans", 11, QFont.Bold))
        layout.addWidget(self.title)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def show_node(self, name: str, G: nx.DiGraph, model):
        if name not in G.nodes:
            return
        data = G.nodes[name]
        node_ref = data.get("node_ref")
        self.title.setText(f"Node: {name}")
        lines = [f"Op Type:  {data.get('op_type', '?')}"]
        lines.append(f"Name:     {name}")

        if node_ref:
            lines.append("")
            lines.append("Inputs:")
            all_vi = {vi.name: vi for vi in model.graph.value_info}
            all_vi.update({vi.name: vi for vi in model.graph.input})
            all_vi.update({vi.name: vi for vi in model.graph.output})
            init_names = {ini.name for ini in model.graph.initializer}
            for inp in node_ref.input:
                flag = " [weight]" if inp in init_names else ""
                shape = tensor_shape_str(all_vi[inp].type) if inp in all_vi else "?"
                lines.append(f"  • {inp}{flag}  {shape}")

            lines.append("")
            lines.append("Outputs:")
            for out in node_ref.output:
                shape = tensor_shape_str(all_vi[out].type) if out in all_vi else "?"
                lines.append(f"  • {out}  {shape}")

            if node_ref.attribute:
                lines.append("")
                lines.append("Attributes:")
                for attr in node_ref.attribute:
                    lines.append(f"  {attr.name}: {_attr_value(attr)}")
        self.text.setPlainText("\n".join(lines))

    def clear(self):
        self.title.setText("Select a node")
        self.text.clear()


def _attr_value(attr) -> str:
    """Pretty-print an ONNX attribute."""
    t = attr.type
    if t == onnx.AttributeProto.FLOAT:   return str(attr.f)
    if t == onnx.AttributeProto.INT:     return str(attr.i)
    if t == onnx.AttributeProto.STRING:  return attr.s.decode("utf-8")
    if t == onnx.AttributeProto.FLOATS:  return str(list(attr.floats)[:8])
    if t == onnx.AttributeProto.INTS:    return str(list(attr.ints)[:8])
    if t == onnx.AttributeProto.GRAPH:   return "<graph>"
    if t == onnx.AttributeProto.TENSOR:
        tp = attr.t
        arr = np.array(tp.float_data or tp.int32_data or tp.int64_data)
        return f"Tensor shape={list(tp.dims)}"
    return str(attr)[:60]


# ──────────────────────────────────────────────────────────────────────────────
# Model info panel
# ──────────────────────────────────────────────────────────────────────────────

class ModelInfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def show_model(self, model, path: str):
        lines = []
        lines.append(f"File:          {path}")
        lines.append(f"IR version:    {model.ir_version}")
        opset_str = ", ".join((o.domain or "ai.onnx") + ":" + str(o.version) for o in model.opset_import)
        lines.append(f"Opset:         [{opset_str}]")
        lines.append(f"Producer:      {model.producer_name or '?'} {model.producer_version or ''}")
        lines.append(f"Domain:        {model.domain or '(default)'}")
        lines.append(f"Model version: {model.model_version}")
        lines.append(f"Doc string:    {model.doc_string or '(none)'}")
        lines.append("")
        lines.append(f"Nodes:         {len(model.graph.node)}")
        op_counts = collections.Counter(n.op_type for n in model.graph.node)
        for op, cnt in op_counts.most_common():
            lines.append(f"  {op:<25} ×{cnt}")
        lines.append("")
        lines.append("Graph inputs:")
        for inp in model.graph.input:
            shape = tensor_shape_str(inp.type)
            dtype = dtype_name(inp.type.tensor_type.elem_type)
            lines.append(f"  • {inp.name}  {dtype} {shape}")
        lines.append("")
        lines.append("Graph outputs:")
        for out in model.graph.output:
            shape = tensor_shape_str(out.type)
            dtype = dtype_name(out.type.tensor_type.elem_type)
            lines.append(f"  • {out.name}  {dtype} {shape}")
        lines.append("")
        init_bytes = sum(len(i.raw_data) for i in model.graph.initializer)
        lines.append(f"Weights:       {len(model.graph.initializer)} tensors  ({init_bytes/1e6:.2f} MB)")
        self.text.setPlainText("\n".join(lines))

    def clear(self):
        self.text.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Legend widget
# ──────────────────────────────────────────────────────────────────────────────

class LegendWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        title = QLabel("Op-type legend")
        title.setFont(QFont("Sans", 9, QFont.Bold))
        layout.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(2)
        for op, color in sorted(OP_COLORS.items()):
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
            label = QLabel(op)
            label.setFont(QFont("monospace", 8))
            row.addWidget(swatch)
            row.addWidget(label)
            row.addStretch()
            inner_layout.addLayout(row)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)


# ──────────────────────────────────────────────────────────────────────────────
# Cut-edges panel  — shows selected tensor cuts and allows clearing them
# ──────────────────────────────────────────────────────────────────────────────

class CutEdgesPanel(QWidget):
    """Right-panel tab that lists the tensors selected as cut boundaries."""

    clearRequested  = pyqtSignal()          # user clicked "Clear all"
    removeTensor    = pyqtSignal(str)       # user removed a single tensor

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Selected cut edges")
        title.setFont(QFont("Sans", 11, QFont.Bold))
        layout.addWidget(title)

        hint = QLabel(
            "Click any edge (connection) in the graph to mark it for cutting.\n"
            "Orange dashed = will be cut.  Click again to deselect."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999; font-size: 10px;")
        layout.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setSpacing(2)
        self._inner_layout.addStretch()
        self.scroll.setWidget(self._inner)
        layout.addWidget(self.scroll, 1)

        self._count_label = QLabel("0 edges selected")
        self._count_label.setStyleSheet("font-size: 10px; color: #9cdcfe;")
        layout.addWidget(self._count_label)

        btn_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview split")
        self._preview_btn.setEnabled(False)
        self._split_btn   = QPushButton("✂  Split now")
        self._split_btn.setEnabled(False)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clearRequested)
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._split_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        self._rows: dict = {}   # tensor_name -> QWidget row

    # ── public API ────────────────────────────────────────────────────────────

    def add_tensor(self, tensor: str):
        if tensor in self._rows or not tensor:
            return
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(2, 0, 2, 0)
        icon = QLabel("✂")
        icon.setFixedWidth(16)
        lbl = QLabel(tensor if len(tensor) <= 36 else tensor[:34] + "…")
        lbl.setFont(QFont("Monospace", 8))
        lbl.setToolTip(tensor)
        rm = QPushButton("✕")
        rm.setFixedSize(20, 20)
        rm.setStyleSheet("QPushButton { background: #555; color: white; padding: 0; }")
        rm.clicked.connect(lambda _, t=tensor: self.removeTensor.emit(t))
        row_l.addWidget(icon)
        row_l.addWidget(lbl, 1)
        row_l.addWidget(rm)
        # insert before the trailing stretch
        self._inner_layout.insertWidget(self._inner_layout.count() - 1, row_w)
        self._rows[tensor] = row_w
        self._refresh_count()

    def remove_tensor(self, tensor: str):
        w = self._rows.pop(tensor, None)
        if w:
            self._inner_layout.removeWidget(w)
            w.deleteLater()
        self._refresh_count()

    def clear(self):
        for w in list(self._rows.values()):
            self._inner_layout.removeWidget(w)
            w.deleteLater()
        self._rows.clear()
        self._refresh_count()

    def tensors(self) -> list:
        return list(self._rows.keys())

    def _refresh_count(self):
        n = len(self._rows)
        self._count_label.setText(f"{n} edge{'s' if n != 1 else ''} selected")
        self._preview_btn.setEnabled(n > 0)
        self._split_btn.setEnabled(n > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ONNX Model Visualizer & Splitter")
        self.resize(1600, 950)

        self.model = None
        self.model_path = ""
        self.G = None
        self.pos = {}
        self.dark_mode = True

        self._build_ui()
        self._apply_theme()
        self.setAcceptDrops(True)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready — load an ONNX model (File toolbar or drag & drop a .onnx file).")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        self.addToolBar(tb)

        a_load = QAction("📂  Load Model", self)
        a_load.setShortcut("Ctrl+O")
        a_load.triggered.connect(self.load_model)
        tb.addAction(a_load)

        tb.addSeparator()

        a_split = QAction("✂️  Split Model", self)
        a_split.setShortcut("Ctrl+Shift+S")
        a_split.triggered.connect(self.open_split_dialog)
        tb.addAction(a_split)

        tb.addSeparator()

        a_export = QAction("🖼  Export Graph PNG", self)
        a_export.triggered.connect(self.export_png)
        tb.addAction(a_export)

        a_validate = QAction("✅  Validate Model", self)
        a_validate.triggered.connect(self.validate_model)
        tb.addAction(a_validate)

        tb.addSeparator()

        a_theme = QAction("🌙  Toggle Theme", self)
        a_theme.triggered.connect(self.toggle_theme)
        tb.addAction(a_theme)

        # ── central area ─────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: node tree + legend
        left = QWidget()
        left.setFixedWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        search_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search nodes…")
        self.search_box.textChanged.connect(self._filter_tree)
        search_row.addWidget(self.search_box)
        left_layout.addLayout(search_row)

        self.node_tree = QTreeWidget()
        self.node_tree.setHeaderLabel("Nodes")
        self.node_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.node_tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.node_tree, 3)

        self.legend = LegendWidget()
        left_layout.addWidget(self.legend, 2)

        splitter.addWidget(left)

        # Center: graph canvas + nav toolbar
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = GraphCanvas(self)
        self.canvas.nodeClicked.connect(self._on_canvas_node_clicked)
        self.canvas.setMinimumHeight(400)

        # zoom hint bar
        zoom_bar = QHBoxLayout()
        zoom_bar.addWidget(QLabel("Scroll: zoom  |  Drag: pan  |  Ctrl+0: fit  |  Click node: select"))
        zoom_bar.addStretch()
        zoom_in_btn  = QPushButton("+")
        zoom_out_btn = QPushButton("−")
        fit_btn      = QPushButton("Fit")
        zoom_in_btn.setFixedWidth(30)
        zoom_out_btn.setFixedWidth(30)
        fit_btn.setFixedWidth(42)
        zoom_in_btn.clicked.connect(lambda: self.canvas.scale(1.25, 1.25))
        zoom_out_btn.clicked.connect(lambda: self.canvas.scale(1/1.25, 1/1.25))
        fit_btn.clicked.connect(lambda: self.canvas.fitInView(
            self.canvas._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40),
            Qt.KeepAspectRatio))
        zoom_bar.addWidget(zoom_out_btn)
        zoom_bar.addWidget(zoom_in_btn)
        zoom_bar.addWidget(fit_btn)
        center_layout.addLayout(zoom_bar)
        center_layout.addWidget(self.canvas)

        # split preview button
        self.split_preview_btn = QPushButton("Preview split at selected node")
        self.split_preview_btn.setEnabled(False)
        self.split_preview_btn.clicked.connect(self._preview_split)
        center_layout.addWidget(self.split_preview_btn)

        splitter.addWidget(center)

        # Right panel: tabs
        right_tabs = QTabWidget()
        right_tabs.setFixedWidth(360)

        self.node_info = NodeInfoPanel()
        right_tabs.addTab(self.node_info, "Node Info")

        self.model_info = ModelInfoPanel()
        right_tabs.addTab(self.model_info, "Model Info")

        self.cut_panel = CutEdgesPanel()
        right_tabs.addTab(self.cut_panel, "✂ Cut Edges")
        # wire cut-panel buttons
        self.cut_panel._preview_btn.clicked.connect(self._preview_split)
        self.cut_panel._split_btn.clicked.connect(self._split_by_cut_edges)
        self.cut_panel.clearRequested.connect(self._clear_cut_edges)
        self.cut_panel.removeTensor.connect(self._remove_cut_tensor)
        # wire canvas edge-cut toggles into the panel
        self.canvas.edgeCutToggled.connect(self._on_edge_cut_toggled)

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(1, 1)

    # ── theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(DARK_STYLE if self.dark_mode else LIGHT_STYLE)
        if self.G:
            # Redraw but preserve any edge cuts the user has selected
            self.canvas.draw_graph(self.G, self.pos, dark=self.dark_mode,
                                   preserve_cuts=True)
            self.canvas._refresh_highlights()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    # ── loading ───────────────────────────────────────────────────────────────

    def load_model(self):
        if not ONNX_AVAILABLE:
            QMessageBox.critical(self, "ONNX not found",
                                 "Please install onnx:\n  pip install onnx")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ONNX Model", "", "ONNX Models (*.onnx);;All Files (*)"
        )
        if not path:
            return
        self._load_model_from_path(path)

    def _load_model_from_path(self, path: str):
        """Start loading an ONNX model from a file path (used by both dialog and drag-drop)."""
        if not ONNX_AVAILABLE:
            QMessageBox.critical(self, "ONNX not found",
                                 "Please install onnx:\n  pip install onnx")
            return
        self.statusBar().showMessage(f"Loading {path}…")
        pb = QProgressBar()
        pb.setRange(0, 0)  # indeterminate
        self.statusBar().addWidget(pb)

        self._worker = LoadWorker(path)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(lambda m, e: self._on_model_loaded(m, e, path, pb))
        self._worker.start()

    # ── drag-and-drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                if url.toLocalFile().lower().endswith(".onnx"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".onnx"):
                self._load_model_from_path(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def _on_model_loaded(self, model, error, path, pb):
        self.statusBar().removeWidget(pb)
        if error:
            QMessageBox.warning(self, "Load error",
                                f"Model loaded with warnings:\n{error}")
        if model is None:
            self.statusBar().showMessage("Failed to load model.")
            return
        self.model = model
        self.model_path = path
        self.setWindowTitle(f"ONNX Visualizer — {Path(path).name}")
        self._build_graph()
        self.model_info.show_model(model, path)
        self.node_info.clear()
        self.split_preview_btn.setEnabled(False)
        self.statusBar().showMessage(
            f"Loaded {Path(path).name}  |  {len(model.graph.node)} nodes"
        )

    def _build_graph(self):
        self.G = build_graph(self.model)
        self.statusBar().showMessage("Computing layout…")
        QApplication.processEvents()
        self.pos = hierarchical_layout(self.G)
        self._populate_tree()
        self.canvas.draw_graph(self.G, self.pos, dark=self.dark_mode)

    def _populate_tree(self):
        self.node_tree.clear()
        if self.G is None:
            return
        # group by op_type, preserving graph order within each group
        groups = collections.defaultdict(list)
        for n in self.G.nodes:  # nx preserves insertion order (topological)
            op = self.G.nodes[n].get("op_type", "Unknown")
            groups[op].append(n)
        for op in sorted(groups):
            parent = QTreeWidgetItem(self.node_tree, [f"{op} ({len(groups[op])})"])
            parent.setData(0, Qt.UserRole, None)
            for n in groups[op]:
                child = QTreeWidgetItem(parent, [n])
                child.setData(0, Qt.UserRole, n)
            parent.setExpanded(False)

    def _filter_tree(self, text: str):
        text = text.lower()
        for i in range(self.node_tree.topLevelItemCount()):
            grp = self.node_tree.topLevelItem(i)
            visible_children = 0
            for j in range(grp.childCount()):
                child = grp.child(j)
                match = text in child.text(0).lower()
                child.setHidden(not match)
                if match:
                    visible_children += 1
            grp.setHidden(visible_children == 0)
            if text and visible_children > 0:
                grp.setExpanded(True)

    def _on_tree_item_clicked(self, item, _col):
        name = item.data(0, Qt.UserRole)
        if name and self.G:
            self.canvas.select_node(name)
            self.node_info.show_node(name, self.G, self.model)
            self.split_preview_btn.setEnabled(True)
            self._selected_node = name

    def _on_canvas_node_clicked(self, name: str):
        self.node_info.show_node(name, self.G, self.model)
        self.split_preview_btn.setEnabled(True)
        self._selected_node = name
        self.statusBar().showMessage(
            f"Selected node: {name}  [{self.G.nodes[name].get('op_type', '')}]"
        )
        # also highlight in tree
        self._highlight_tree_node(name)

    def _highlight_tree_node(self, name: str):
        for i in range(self.node_tree.topLevelItemCount()):
            grp = self.node_tree.topLevelItem(i)
            for j in range(grp.childCount()):
                child = grp.child(j)
                if child.data(0, Qt.UserRole) == name:
                    self.node_tree.setCurrentItem(child)
                    grp.setExpanded(True)
                    return

    def _preview_split(self):
        # prefer edge-cut preview if any cuts selected
        cut_tensors = self.canvas.get_cut_tensors()
        if cut_tensors:
            self.canvas.highlight_split_preview(cut_tensors=cut_tensors)
            self.statusBar().showMessage(
                f"Split preview: blue = part 1, red = part 2  "
                f"({len(cut_tensors)} cut edge{'s' if len(cut_tensors)!=1 else ''})"
            )
        elif hasattr(self, "_selected_node") and self._selected_node:
            self.canvas.highlight_split_preview(split_node=self._selected_node)
            self.statusBar().showMessage(
                f"Split preview: blue = part 1 (up to {self._selected_node}), red = part 2"
            )
        else:
            self.statusBar().showMessage(
                "Select a node or click edge(s) to preview a split."
            )

    # ── split ─────────────────────────────────────────────────────────────────

    # ── cut-edge callbacks ────────────────────────────────────────────────────

    def _on_edge_cut_toggled(self, tensor: str, is_cut: bool):
        """Synchronise canvas edge state ↔ CutEdgesPanel."""
        if not tensor:                    # empty tensor = clear signal
            self.cut_panel.clear()
            return
        if is_cut:
            self.cut_panel.add_tensor(tensor)
        else:
            self.cut_panel.remove_tensor(tensor)
        n = len(self.canvas.get_cut_tensors())
        self.statusBar().showMessage(
            f"{n} edge cut{'s' if n != 1 else ''} selected"
            + ("  — click 'Preview split' or 'Split now'" if n > 0 else "")
        )

    def _remove_cut_tensor(self, tensor: str):
        """Called when user clicks ✕ next to a tensor in CutEdgesPanel."""
        self.canvas.set_cut_tensor(tensor, False)
        self.cut_panel.remove_tensor(tensor)

    def _clear_cut_edges(self):
        self.canvas.clear_cut_tensors()
        self.cut_panel.clear()
        self.canvas.clear_highlight()
        self.statusBar().showMessage("Cut edges cleared.")

    def _split_by_cut_edges(self):
        """Perform the edge-cut split using currently selected cut tensors."""
        if self.model is None:
            QMessageBox.information(self, "No model", "Load an ONNX model first.")
            return
        cut_tensors = self.canvas.get_cut_tensors()
        if not cut_tensors:
            QMessageBox.information(self, "No cuts",
                                    "Click edges in the graph to mark them for cutting.")
            return
        stem = Path(self.model_path).stem if self.model_path else "model"
        out_dir = str(Path(self.model_path).parent) if self.model_path else os.getcwd()
        safe = re.sub(r"[^\w]", "_",
                      "cut_" + "_".join(sorted(t[:12] for t in cut_tensors))[:40])
        default_p1 = os.path.join(out_dir, f"{stem}__part1__{safe}.onnx")
        default_p2 = os.path.join(out_dir, f"{stem}__part2__{safe}.onnx")
        # Confirmation dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm edge-cut split")
        edges_str = "\n".join(f"  • {t}" for t in sorted(cut_tensors))
        msg.setText(
            f"Split at {len(cut_tensors)} selected edge(s):\n{edges_str}\n\n"
            f"Output directory: {out_dir}"
        )
        msg.setInformativeText(
            f"Part 1: {Path(default_p1).name}\n"
            f"Part 2: {Path(default_p2).name}"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if msg.exec_() != QMessageBox.Ok:
            return
        self._do_edge_split(list(cut_tensors), default_p1, default_p2)

    def _do_edge_split(self, cut_tensors: list, p1_path: str, p2_path: str):
        try:
            p1, p2 = split_model_at_tensors(self.model, cut_tensors)
            onnx.save(p1, p1_path)
            onnx.save(p2, p2_path)
            msg = (
                f"Edge-cut split complete!\n\n"
                f"Cut edges ({len(cut_tensors)}): "
                + ", ".join(sorted(cut_tensors)[:5])
                + ("..." if len(cut_tensors) > 5 else "")
                + f"\n\nPart 1: {p1_path}\n  Nodes: {len(p1.graph.node)}\n"
                  f"  Outputs: {len(p1.graph.output)}\n\n"
                  f"Part 2: {p2_path}\n  Nodes: {len(p2.graph.node)}\n"
                  f"  Inputs: {len(p2.graph.input)}"
            )
            QMessageBox.information(self, "Split complete", msg)
            self.statusBar().showMessage(
                f"Edge-cut split done — part1: {len(p1.graph.node)} nodes, "
                f"part2: {len(p2.graph.node)} nodes"
            )
        except Exception as e:
            QMessageBox.critical(self, "Split failed", str(e))

    # ── split (node-based, via dialog) ────────────────────────────────────────

    def open_split_dialog(self):
        """Open the node-based split dialog, or use edge cuts if any are selected."""
        if self.model is None:
            QMessageBox.information(self, "No model", "Load an ONNX model first.")
            return
        cut_tensors = self.canvas.get_cut_tensors()
        if cut_tensors:
            # shortcut: user already has edges marked — go straight to edge-cut
            self._split_by_cut_edges()
            return
        node_names = [
            node_name(n, idx) for idx, n in enumerate(self.model.graph.node)
        ]
        dlg = SplitDialog(node_names, self.model_path, self)
        if hasattr(self, "_selected_node") and self._selected_node:
            idx = dlg.node_combo.findText(self._selected_node)
            if idx >= 0:
                dlg.node_combo.setCurrentIndex(idx)
        if dlg.exec_() != QDialog.Accepted:
            return
        split_name, p1_path, p2_path = dlg.get_params()
        self._do_node_split(split_name, p1_path, p2_path)

    def _do_node_split(self, split_name: str, p1_path: str, p2_path: str):
        try:
            p1, p2 = split_model_at_node(self.model, split_name)
            onnx.save(p1, p1_path)
            onnx.save(p2, p2_path)
            msg = (
                f"Model split successfully!\n\n"
                f"Part 1: {p1_path}\n"
                f"  Nodes: {len(p1.graph.node)}\n\n"
                f"Part 2: {p2_path}\n"
                f"  Nodes: {len(p2.graph.node)}"
            )
            QMessageBox.information(self, "Split complete", msg)
            self.statusBar().showMessage(
                f"Split done — part1: {len(p1.graph.node)} nodes, "
                f"part2: {len(p2.graph.node)} nodes"
            )
        except Exception as e:
            QMessageBox.critical(self, "Split failed", str(e))


    # ── validate ──────────────────────────────────────────────────────────────

    def validate_model(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Load an ONNX model first.")
            return
        try:
            onnx.checker.check_model(self.model)
            QMessageBox.information(self, "Validation", "✅  Model is valid.")
        except Exception as e:
            QMessageBox.warning(self, "Validation failed", str(e))

    # ── export ────────────────────────────────────────────────────────────────

    def export_png(self):
        if self.G is None:
            QMessageBox.information(self, "No graph", "Load an ONNX model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export graph as PNG", "", "PNG Images (*.png)"
        )
        if path:
            self.canvas.export_to_png(path)
            self.statusBar().showMessage(f"Graph exported to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not ONNX_AVAILABLE:
        print("ERROR: onnx package not found. Install it with:  pip install onnx")
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setApplicationName("ONNX Visualizer & Splitter")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
