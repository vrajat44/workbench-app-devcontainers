"""
Lineage Graph Builder
Converts Provenance DataFrames into interactive Pyvis network graphs.
All graphs use left-to-right hierarchical layout: source → activity → target.
Provenance resources are excluded; only clinical/administrative resources appear.
"""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

# ── Color Palette ─────────────────────────────────────────────────────────────

RESOURCE_COLORS = {
    "Patient": "#4285F4",
    "Observation": "#34A853",
    "Condition": "#EA4335",
    "Procedure": "#FBBC04",
    "MedicationRequest": "#9C27B0",
    "Encounter": "#FF6D00",
    "Task": "#607D8B",
    "CarePlan": "#00BCD4",
    "RequestGroup": "#795548",
    "Basic": "#9E9E9E",
    "VerificationResult": "#E91E63",
    "QuestionnaireResponse": "#3F51B5",
    "DocumentReference": "#8BC34A",
    "Communication": "#FF5722",
    "CommunicationRequest": "#FF9800",
    "Device": "#CDDC39",
    "Binary": "#B0BEC5",
    "Person": "#2196F3",
    "Organization": "#673AB7",
    "Group": "#009688",
    "CoverageEligibilityResponse": "#F48FB1",
    "CoverageEligibilityRequest": "#CE93D8",
    "AuditEvent": "#B0BEC5",
    "Consent": "#80DEEA",
    "Coverage": "#A5D6A7",
    "InsurancePlan": "#FFE082",
    "ResearchSubject": "#BCAAA4",
}

ACTIVITY_COLORS = {
    "INGEST": "#1565C0",
    "Action Engine": "#2E7D32",
    "enrichment:fhir-data-quality-operational": "#C62828",
    "enrichment:fhir-data-quality": "#D32F2F",
    "enrichment:patient-priority": "#6A1B9A",
    "enrichment:participant-index": "#4527A0",
    "enrichment:targeted-eligibility": "#00838F",
    "workflow:apply-workflow": "#E65100",
    "timer-action": "#33691E",
    "Enrollment": "#0277BD",
    "Third party account creation": "#558B2F",
}

DEFAULT_RESOURCE_COLOR = "#78909C"
DEFAULT_ACTIVITY_COLOR = "#455A64"


def _get_resource_color(rtype: str) -> str:
    return RESOURCE_COLORS.get(rtype, DEFAULT_RESOURCE_COLOR)


def _get_activity_color(activity: str) -> str:
    for key, color in ACTIVITY_COLORS.items():
        if key in activity:
            return color
    return DEFAULT_ACTIVITY_COLOR


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _apply_hierarchical_lr(net):
    """Apply left-to-right hierarchical layout options to a pyvis Network."""
    net.set_options(json.dumps({
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "nodeSpacing": 150,
                "levelSeparation": 280,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 180,
                "centralGravity": 0.0,
                "springLength": 200,
                "springConstant": 0.01,
                "damping": 0.09,
            },
            "enabled": True,
            "stabilization": {
                "enabled": True,
                "iterations": 200,
            },
        },
        "edges": {
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
                "roundness": 0.4,
            },
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
        },
        "interaction": {
            "dragNodes": True,
            "zoomView": True,
            "dragView": True,
            "hover": True,
            "tooltipDelay": 100,
        },
    }))


# ── High-Level Lineage Graph ─────────────────────────────────────────────────


def build_high_level_graph(df: pd.DataFrame) -> str:
    """
    Build an interactive left-to-right graph from the high-level lineage DataFrame.
    Layout: Source Resources (L0) → Activities (L1) → Target Resources (L2).
    Provenance resources are already filtered out at the query level.

    df columns: entity_type, activity, target_type, cnt
    """
    from pyvis.network import Network

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#fafafa",
        font_color="#333",
        select_menu=False,
        filter_menu=False,
    )
    _apply_hierarchical_lr(net)

    # Aggregate node sizes
    entity_totals: dict[str, int] = {}
    target_totals: dict[str, int] = {}
    activity_totals: dict[str, int] = {}

    for _, row in df.iterrows():
        etype = row["entity_type"]
        activity = row["activity"]
        ttype = row["target_type"]
        cnt = int(row["cnt"])

        if etype not in ("(none)", "Provenance"):
            entity_totals[etype] = entity_totals.get(etype, 0) + cnt
        if ttype not in ("(none)", "Provenance"):
            target_totals[ttype] = target_totals.get(ttype, 0) + cnt
        activity_totals[activity] = activity_totals.get(activity, 0) + cnt

    all_resource_totals = {}
    for k, v in entity_totals.items():
        all_resource_totals[k] = all_resource_totals.get(k, 0) + v
    for k, v in target_totals.items():
        all_resource_totals[k] = all_resource_totals.get(k, 0) + v
    max_resource = max(all_resource_totals.values()) if all_resource_totals else 1

    # Level 0: Source entity resource types (left side)
    for rtype, total in entity_totals.items():
        size = max(15, min(60, 15 + 45 * (total / max_resource)))
        net.add_node(
            f"src:{rtype}",
            label=rtype,
            title=f"Source: {rtype}\nTotal flow: {_format_count(total)}",
            color=_get_resource_color(rtype),
            size=size,
            shape="dot",
            font={"size": 14, "face": "Arial"},
            level=0,
        )

    # Level 1: Activity nodes (middle)
    max_activity = max(activity_totals.values()) if activity_totals else 1
    for activity, total in activity_totals.items():
        size = max(10, min(40, 10 + 30 * (total / max_activity)))
        label = activity if len(activity) < 25 else activity[:22] + "..."
        net.add_node(
            f"act:{activity}",
            label=label,
            title=f"Activity: {activity}\nTotal: {_format_count(total)}",
            color=_get_activity_color(activity),
            size=size,
            shape="diamond",
            font={"size": 11, "face": "Arial", "color": "#fff"},
            level=1,
        )

    # Level 2: Target resource types (right side)
    for rtype, total in target_totals.items():
        size = max(15, min(60, 15 + 45 * (total / max_resource)))
        net.add_node(
            f"tgt:{rtype}",
            label=rtype,
            title=f"Target: {rtype}\nTotal flow: {_format_count(total)}",
            color=_get_resource_color(rtype),
            size=size,
            shape="dot",
            font={"size": 14, "face": "Arial"},
            level=2,
        )

    # Edges: source → activity → target
    node_ids = set(n["id"] for n in net.nodes)
    edge_agg: dict[tuple, int] = {}

    for _, row in df.iterrows():
        etype = row["entity_type"]
        activity = row["activity"]
        ttype = row["target_type"]
        cnt = int(row["cnt"])

        act_node = f"act:{activity}"
        target_node = f"tgt:{ttype}"

        if target_node not in node_ids or act_node not in node_ids:
            continue

        # Source → Activity
        if etype not in ("(none)", "Provenance"):
            src_node = f"src:{etype}"
            if src_node in node_ids:
                key = (src_node, act_node)
                edge_agg[key] = edge_agg.get(key, 0) + cnt

        # Activity → Target
        key = (act_node, target_node)
        edge_agg[key] = edge_agg.get(key, 0) + cnt

    for (src, dst), cnt in edge_agg.items():
        width = max(1, min(8, 1 + 7 * (cnt / max_resource)))
        net.add_edge(
            src, dst,
            value=cnt,
            title=f"{src.split(':', 1)[1]} → {dst.split(':', 1)[1]}: {_format_count(cnt)}",
            color={"color": "#aaa", "opacity": 0.6},
            width=width,
        )

    # Generate HTML
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    net.save_graph(tmp.name)
    tmp.close()
    html_content = Path(tmp.name).read_text()

    custom_css = """
    <style>
      body { margin: 0; padding: 0; }
      #mynetwork { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
    """
    html_content = html_content.replace("</head>", custom_css + "</head>")
    return html_content


# ── Instance-Level Lineage Graph ─────────────────────────────────────────────


def build_instance_graph(
    df: pd.DataFrame,
    selected_type: str,
    selected_id: str,
) -> str:
    """
    Build an interactive left-to-right instance lineage graph.
    Sources on the left, selected target on the far right.
    Provenance resources are already filtered at the query level.

    df columns: provenance_id, activity, target_type, target_id,
                entity_role, entity_type, entity_id, recorded, depth
    """
    from pyvis.network import Network

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#fafafa",
        font_color="#333",
        select_menu=False,
        filter_menu=False,
    )
    _apply_hierarchical_lr(net)

    added_nodes = set()
    added_edges = set()

    max_depth = int(df["depth"].max()) if not df.empty and "depth" in df.columns else 1

    def depth_to_level(depth: int) -> int:
        """Higher depth = further back = further left (lower level number)."""
        return (max_depth - depth) * 2

    def add_resource_node(rtype, rid, is_selected=False, depth=None):
        node_id = f"{rtype}/{rid}"
        if node_id in added_nodes:
            return node_id
        added_nodes.add(node_id)

        short_id = rid[:8] + "..." if len(rid) > 12 else rid
        label = f"{rtype}\n{short_id}"

        if is_selected:
            level = max_depth * 2 + 1
        elif depth is not None:
            level = depth_to_level(depth)
        else:
            level = 0

        if is_selected:
            net.add_node(
                node_id,
                label=label,
                title=f"🎯 SELECTED: {rtype}/{rid}",
                color={"background": _get_resource_color(rtype), "border": "#FFD700"},
                size=40,
                shape="dot",
                borderWidth=4,
                font={"size": 14, "face": "Arial", "bold": True},
                level=level,
            )
        else:
            net.add_node(
                node_id,
                label=label,
                title=f"{rtype}/{rid}",
                color=_get_resource_color(rtype),
                size=25,
                shape="dot",
                font={"size": 12, "face": "Arial"},
                level=level,
            )
        return node_id

    def add_activity_node(prov_id, activity, recorded, depth):
        node_id = f"prov:{prov_id}"
        if node_id in added_nodes:
            return node_id
        added_nodes.add(node_id)

        label = activity if len(activity) < 20 else activity[:17] + "..."
        ts = str(recorded)[:19] if recorded else ""
        level = depth_to_level(depth) + 1

        net.add_node(
            node_id,
            label=label,
            title=f"Activity: {activity}\nProvenance: {prov_id}\nRecorded: {ts}\nDepth: {depth}",
            color=_get_activity_color(activity),
            size=18,
            shape="diamond",
            font={"size": 10, "face": "Arial", "color": "#fff"},
            level=level,
        )
        return node_id

    # Add the selected (final) node
    add_resource_node(selected_type, selected_id, is_selected=True)

    for _, row in df.iterrows():
        prov_id = row["provenance_id"]
        activity = row["activity"]
        target_type = row["target_type"]
        target_id = row.get("target_id", "")
        entity_type = row.get("entity_type", "(none)")
        entity_id = row.get("entity_id", None)
        recorded = row.get("recorded", "")
        depth = int(row.get("depth", 1))

        if not target_id:
            continue

        # Skip any Provenance that slipped through
        if target_type == "Provenance" or entity_type == "Provenance":
            continue

        act_node = add_activity_node(prov_id, activity, recorded, depth)

        target_node = add_resource_node(
            target_type, target_id,
            is_selected=(target_type == selected_type and target_id == selected_id),
            depth=depth - 1 if depth > 1 else 0,
        )

        edge_key = (act_node, target_node)
        if edge_key not in added_edges:
            added_edges.add(edge_key)
            net.add_edge(
                act_node,
                target_node,
                title=f"{activity} → {target_type}/{target_id[:8]}...",
                color={"color": "#666"},
                width=2,
            )

        if entity_id and entity_type not in ("(none)", "Provenance"):
            entity_node = add_resource_node(entity_type, entity_id, depth=depth)
            edge_key = (entity_node, act_node)
            if edge_key not in added_edges:
                added_edges.add(edge_key)
                role = row.get("entity_role", "source")
                net.add_edge(
                    entity_node,
                    act_node,
                    title=f"{entity_type}/{entity_id[:8]}... →[{role}]→ {activity}",
                    color={"color": "#999"},
                    width=1.5,
                    dashes=role == "quotation",
                )

    if not added_nodes:
        return "<div style='padding:40px;text-align:center;color:#666;'><h3>No lineage data found for this resource.</h3></div>"

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    net.save_graph(tmp.name)
    tmp.close()
    html_content = Path(tmp.name).read_text()

    legend_html = _build_legend(df, selected_type, selected_id)
    html_content = html_content.replace("</body>", legend_html + "</body>")

    custom_css = """
    <style>
      body { margin: 0; padding: 0; }
      #mynetwork { border: 1px solid #e0e0e0; border-radius: 8px; }
      .lineage-legend {
        position: absolute; top: 10px; right: 10px;
        background: rgba(255,255,255,0.95); border: 1px solid #ddd;
        border-radius: 8px; padding: 12px; font-family: Arial;
        font-size: 12px; max-width: 250px; z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      .lineage-legend h4 { margin: 0 0 8px 0; font-size: 13px; }
      .legend-item { display: flex; align-items: center; margin: 3px 0; }
      .legend-dot { width: 12px; height: 12px; border-radius: 50%;
                    margin-right: 8px; flex-shrink: 0; }
      .legend-diamond { width: 12px; height: 12px; transform: rotate(45deg);
                        margin-right: 8px; flex-shrink: 0; }
    </style>
    """
    html_content = html_content.replace("</head>", custom_css + "</head>")
    return html_content


def _build_legend(df: pd.DataFrame, selected_type: str, selected_id: str) -> str:
    """Build HTML legend for the instance graph."""
    resource_types = set()
    activities = set()

    resource_types.add(selected_type)
    for _, row in df.iterrows():
        tt = row.get("target_type")
        if tt and tt not in ("(none)", "Provenance"):
            resource_types.add(tt)
        et = row.get("entity_type")
        if et and et not in ("(none)", "Provenance"):
            resource_types.add(et)
        if row.get("activity"):
            activities.add(row["activity"])

    items = []
    items.append("<h4>📖 Legend</h4>")
    items.append("<h4>Resource Types</h4>")
    for rt in sorted(resource_types):
        color = _get_resource_color(rt)
        items.append(
            f'<div class="legend-item">'
            f'<div class="legend-dot" style="background:{color}"></div>'
            f'{html.escape(rt)}</div>'
        )

    items.append("<h4 style='margin-top:10px'>Activities</h4>")
    for act in sorted(activities):
        color = _get_activity_color(act)
        label = act if len(act) < 30 else act[:27] + "..."
        items.append(
            f'<div class="legend-item">'
            f'<div class="legend-diamond" style="background:{color}"></div>'
            f'{html.escape(label)}</div>'
        )

    items.append("<div style='margin-top:8px;font-size:11px;color:#888'>← Source &nbsp;&nbsp; Flow direction &nbsp;&nbsp; Target →</div>")

    return f'<div class="lineage-legend">{"".join(items)}</div>'


# ── Simple flat graph (no multi-hop needed) ───────────────────────────────────


def build_flat_instance_graph(
    df: pd.DataFrame,
    selected_type: str,
    selected_id: str,
) -> str:
    """Single-hop lineage graph."""
    if "depth" not in df.columns:
        df = df.copy()
        df["depth"] = 1
    return build_instance_graph(df, selected_type, selected_id)
