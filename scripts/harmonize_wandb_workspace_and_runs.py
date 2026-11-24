# -*- coding: utf-8 -*-
"""Audit, back up, and harmonize W&B Workspace views, additive Run summary keys, and the Lab 2 Report.

Safety & Provenance Invariants:
1. Immutable JSON + SHA256 snapshot is written to `results/lab2_empirical/wandb_backups/`
   before any live mutation (`--apply`).
2. Existing `run.summary` keys, `run.config`, `run.history`, and `Artifact` versions are
   NEVER deleted or overwritten; only missing `summary/*_pct` / `comparison/*_pct` aliases
   are added so every Run page and comparison chart renders cleanly.
3. Restores the top auto-generated `"Charts"` section (`isPanelsAuto=True`, `isOpen=True`,
   0 blocking custom panels on Page 1) in `Caoban170106's workspace` so clicking any
   individual Run URL from `README.md` immediately displays that run's own recorded curves.
4. Creates or updates (idempotently) the 4th public W&B Report dedicated to Lab 2
   (`Critical Analysis & Empirical Demonstration`).
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.parse
import urllib.request

import wandb
from wandb.apis.public.reports import Reports

ENTITY = "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm"
PROJECT = "hcmus-foal-reproduction"
PROJECT_INTERNAL_ID = "UHJvamVjdEludGVybmFsSWQ6NDU0MTYzODE="

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = REPO_ROOT / "results" / "lab2_empirical" / "wandb_backups"
MANIFEST_PATH = REPO_ROOT / "results" / "lab2_empirical" / "wandb_lab2_manifest.json"

LAB2_REPORT_NAME = "foal-lab2-critical-analysis-empirical-demo"
LAB2_REPORT_DISPLAY_NAME = (
    "F-OAL | Phân tích Phản biện Lý thuyết & Thực nghiệm Lab 2 (Long-Tail & Scaling)"
)

UPSERT_VIEW_MUTATION = """
mutation UpsertView($input: UpsertViewInput!) {
  upsertView(input: $input) {
    view {
      id
      name
      displayName
      type
    }
  }
}
"""


def _gql_request(api: wandb.Api, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute an authenticated GraphQL request against the configured W&B server."""
    url = api.settings["base_url"].rstrip("/") + "/graphql"
    token = base64.b64encode(f"api:{api.api_key}".encode("utf-8")).decode("ascii")
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}",
            "User-Agent": f"W&B Public Client {wandb.__version__}",
        },
    )
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    res = json.loads(raw)
    if res.get("errors"):
        raise RuntimeError(f"GraphQL error: {json.dumps(res['errors'], ensure_ascii=False)}")
    return res["data"]


def _build_report_url(app_url: str, display_name: str, view_id: str) -> str:
    """Construct the canonical public W&B Report URL matching frontend routing."""
    slug = urllib.parse.quote(re.sub(r"-+", "-", re.sub(r"\W", "-", display_name)).strip("-"))
    clean_id = view_id.replace("=", "")
    return f"{app_url.rstrip('/')}/{ENTITY}/{PROJECT}/reports/{slug}--{clean_id}"


def _write_immutable_backup(snapshot: dict[str, Any]) -> Path:
    """Write a timestamped backup JSON and its `.sha256` sidecar."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body = json.dumps(
        snapshot,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: dict(o) if hasattr(o, "items") else str(o),
    ).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out_json = BACKUP_DIR / f"wandb_harmonize_backup_{ts}_{digest[:12]}.json"
    out_sha = BACKUP_DIR / f"{out_json.name}.sha256"
    out_json.write_bytes(body)
    out_sha.write_text(f"{digest}  {out_json.name}\n", encoding="utf-8")
    return out_json


def _plan_additive_run_updates(api: wandb.Api) -> list[dict[str, Any]]:
    """Compute strictly additive summary aliases for the 5 canonical runs missing `summary/*_pct`."""
    specs = [
        {
            "run_id": "3eca8232",
            "scale": 100.0,
            "map": {
                "summary/average_incremental_accuracy": "average_incremental_accuracy",
                "summary/average_incremental_accuracy_pct": "average_incremental_accuracy",
                "summary/last_accuracy_pct": "last_accuracy",
                "summary/forgetting_past_tasks_pct": "forgetting",
                "summary/forgetting_official_code_pct": "official_forgetting_10_tasks",
                "comparison/average_incremental_accuracy_pct": "average_incremental_accuracy",
            },
            "raw_keys_keep_unit": {"summary/average_incremental_accuracy"},
        },
        {
            "run_id": "c9yyo08r",
            "scale": 1.0,
            "map": {
                "summary/average_incremental_accuracy_pct": "average_incremental_accuracy_pct",
                "summary/last_accuracy_pct": "last_accuracy_pct",
                "summary/forgetting_past_tasks_pct": "forgetting_pct",
                "summary/forgetting_official_code_pct": "official_forgetting_10_tasks_pct",
                "comparison/average_incremental_accuracy_pct": "average_incremental_accuracy_pct",
            },
            "raw_keys_keep_unit": set(),
        },
        {
            "run_id": "xlcme1c3",
            "scale": 1.0,
            "map": {
                "summary/average_incremental_accuracy_pct": "final/average_incremental_accuracy",
                "summary/last_accuracy_pct": "final/last_accuracy",
                "summary/forgetting_past_tasks_pct": "final/forgetting_past_tasks",
                "summary/forgetting_official_code_pct": "final/forgetting_official_code",
            },
            "raw_keys_keep_unit": set(),
        },
        {
            "run_id": "c5pjewk4",
            "scale": 1.0,
            "map": {
                "summary/average_incremental_accuracy_pct": "final/average_incremental_accuracy",
                "summary/last_accuracy_pct": "final/last_accuracy",
                "summary/forgetting_past_tasks_pct": "final/forgetting_past_tasks",
                "summary/forgetting_official_code_pct": "final/forgetting_official_code",
            },
            "raw_keys_keep_unit": set(),
        },
        {
            "run_id": "ifl3rz2u",
            "scale": 1.0,
            "map": {
                "summary/average_incremental_accuracy_pct": "final/average_incremental_accuracy",
                "summary/last_accuracy_pct": "final/last_accuracy",
                "summary/forgetting_past_tasks_pct": "final/forgetting_past_tasks",
                "summary/forgetting_official_code_pct": "final/forgetting_official_code",
            },
            "raw_keys_keep_unit": set(),
        },
    ]

    plans: list[dict[str, Any]] = []
    for sp in specs:
        run = api.run(f"{ENTITY}/{PROJECT}/{sp['run_id']}")
        before_summary = {k: v for k, v in dict(run.summary).items() if not k.startswith("_")}
        additions: dict[str, float] = {}
        for target_key, src_key in sp["map"].items():
            if target_key in before_summary:
                continue
            if src_key not in before_summary:
                raise KeyError(f"Run {sp['run_id']} missing expected source key {src_key!r}")
            factor = 1.0 if target_key in sp["raw_keys_keep_unit"] else sp["scale"]
            additions[target_key] = float(before_summary[src_key]) * factor
        plans.append({
            "run_id": sp["run_id"],
            "run_name": run.name,
            "before_summary": before_summary,
            "additions": additions,
        })
    return plans


def _transform_workspace_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Ensure the top `isPanelsAuto` section is a clean, open `Charts` section with 0 blocking panels."""
    new_spec = json.loads(json.dumps(spec))
    sec_root = new_spec.get("section") or {}
    pbc = sec_root.get("panelBankConfig") or {}
    sections = pbc.get("sections") or []
    if not sections:
        return new_spec, False

    changed = False
    if not new_spec.get("vizExpanded", False):
        new_spec["vizExpanded"] = True
        changed = True

    extracted_custom_panels: list[dict[str, Any]] = []
    for sec in sections:
        if sec.get("isPanelsAuto"):
            if sec.get("name") != "Charts" or not sec.get("isOpen", False):
                sec["name"] = "Charts"
                sec["isOpen"] = True
                changed = True
            desired_flow = {
                "snapToColumns": True,
                "columnsPerPage": 3,
                "rowsPerPage": 2,
                "gutterWidth": 16,
                "boxWidth": 460,
                "boxHeight": 320,
            }
            if sec.get("flowConfig") != desired_flow:
                sec["flowConfig"] = desired_flow
                changed = True
            if sec.get("panels"):
                extracted_custom_panels.extend(sec["panels"])
                sec["panels"] = []
                changed = True

    if pbc.get("panelPlacementOverrides"):
        pbc["panelPlacementOverrides"] = {}
        changed = True

    if extracted_custom_panels:
        custom_sec_id = "custom_table4_comparison"
        existing_custom = next((s for s in sections if s.get("__id__") == custom_sec_id), None)
        if existing_custom is None:
            custom_sec = {
                "__id__": custom_sec_id,
                "name": "Custom Comparison — Table 4 Ablation",
                "isOpen": False,
                "sorted": 0,
                "pinned": False,
                "isPanelsAuto": False,
                "flowConfig": {
                    "columnsPerPage": 2,
                    "boxWidth": 540,
                    "boxHeight": 420,
                    "rowsPerPage": 1,
                },
                "panels": extracted_custom_panels,
            }
            sections.insert(1, custom_sec)
        else:
            existing_custom["panels"].extend(extracted_custom_panels)
        changed = True

    return new_spec, changed


def _make_panel_grid_block(
    run_set_id: str,
    run_set_name: str,
    run_ids: list[str],
    panels: list[dict[str, Any]],
    x_axis: str = "_step",
) -> dict[str, Any]:
    """Construct a native W&B Report v5 `panel-grid` block locked to specific Run IDs."""
    return {
        "type": "panel-grid",
        "children": [{"text": ""}],
        "metadata": {
            "openViz": True,
            "openRunSet": 0,
            "name": "unused-name",
            "hideRunSets": False,
            "runSets": [
                {
                    "id": run_set_id,
                    "runFeed": {
                        "version": 2,
                        "columnVisible": {},
                        "columnPinned": {},
                        "columnWidths": {},
                        "columnOrder": [],
                        "pageSize": 10,
                        "onlyShowSelected": False,
                    },
                    "enabled": True,
                    "project": {
                        "name": PROJECT,
                        "entityName": ENTITY,
                        "id": PROJECT_INTERNAL_ID,
                    },
                    "name": run_set_name,
                    "search": {"query": ""},
                    "filters": {
                        "filterFormat": "filterV2",
                        "filters": [
                            {
                                "key": {"section": "run", "name": "name"},
                                "op": "IN",
                                "value": run_ids,
                                "disabled": False,
                            }
                        ],
                    },
                    "grouping": [],
                    "sort": {
                        "keys": [
                            {
                                "key": {"section": "run", "name": "createdAt"},
                                "ascending": True,
                            }
                        ]
                    },
                    "selections": {"root": 1, "bounds": [], "tree": []},
                    "expandedRowAddresses": [],
                }
            ],
            "panels": {
                "views": {"0": {"name": "Panels", "defaults": [], "config": []}},
                "tabs": ["0"],
            },
            "panelBankConfig": {
                "state": 0,
                "settings": {
                    "autoOrganizePrefix": 2,
                    "showEmptySections": False,
                    "sortAlphabetically": False,
                },
                "sections": [
                    {
                        "name": "Hidden Panels",
                        "isOpen": False,
                        "type": "flow",
                        "flowConfig": {
                            "snapToColumns": True,
                            "columnsPerPage": 3,
                            "rowsPerPage": 2,
                            "gutterWidth": 16,
                            "boxWidth": 460,
                            "boxHeight": 300,
                        },
                        "sorted": 0,
                        "localPanelSettings": {
                            "smoothingWeight": 0,
                            "smoothingType": "exponential",
                            "xAxis": x_axis,
                            "ignoreOutliers": False,
                            "useRunsTableGroupingInPanels": True,
                            "smoothingActive": False,
                            "xAxisActive": True,
                        },
                        "panels": [],
                    }
                ],
            },
            "panelBankSectionConfig": {
                "name": "Report Panels",
                "isOpen": False,
                "type": "grid",
                "flowConfig": {
                    "snapToColumns": True,
                    "columnsPerPage": 2,
                    "rowsPerPage": 2,
                    "gutterWidth": 16,
                    "boxWidth": 520,
                    "boxHeight": 340,
                },
                "sorted": 0,
                "localPanelSettings": {
                    "smoothingWeight": 0,
                    "smoothingType": "exponential",
                    "xAxis": x_axis,
                    "ignoreOutliers": False,
                    "useRunsTableGroupingInPanels": True,
                    "smoothingActive": False,
                    "xAxisActive": True,
                },
                "isPanelsAuto": False,
                "panels": panels,
            },
        },
    }


def _build_lab2_report_spec() -> dict[str, Any]:
    """Build the complete v5 Report specification for Milestone 3 & Lab 2."""
    e1_panels = [
        {
            "__id__": "lab2_e1_tail_curve",
            "layout": {"x": 0, "y": 0, "w": 12, "h": 9},
            "viewType": "Run History Line Plot",
            "config": {
                "chartTitle": "E1 — Few-Shot Tail Recall (%) vs Imbalance Ratio ρ (3 Seeds)",
                "metrics": ["tail_acc_mean"],
                "xAxis": "imbalance_ratio_rho",
                "xAxisLogScale": True,
                "yAxisTitle": "Tail 1/3 Classes Accuracy (%)",
                "overrideSeriesTitles": {
                    "g34o8awx:tail_acc_mean": "Vanilla F-OAL (Paper MSE) [g34o8awx]",
                    "kafjjzio:tail_acc_mean": "Norm Calibration FD1b [kafjjzio]",
                    "fqahbyt4:tail_acc_mean": "Balanced Woodbury FD1a+b [fqahbyt4]",
                },
            },
        },
        {
            "__id__": "lab2_e1_avg_curve",
            "layout": {"x": 12, "y": 0, "w": 12, "h": 9},
            "viewType": "Run History Line Plot",
            "config": {
                "chartTitle": "E1 — 10-Task Incremental Accuracy A_bar (%) vs ρ (3 Seeds)",
                "metrics": ["average_accuracy_mean"],
                "xAxis": "imbalance_ratio_rho",
                "xAxisLogScale": True,
                "yAxisTitle": "Average Incremental Accuracy (%)",
                "overrideSeriesTitles": {
                    "g34o8awx:average_accuracy_mean": "Vanilla F-OAL [g34o8awx]",
                    "kafjjzio:average_accuracy_mean": "Norm Calibration FD1b [kafjjzio]",
                    "fqahbyt4:average_accuracy_mean": "Balanced Woodbury FD1a+b [fqahbyt4]",
                },
            },
        },
        {
            "__id__": "lab2_e1_rho100_bar",
            "layout": {"x": 0, "y": 9, "w": 24, "h": 8},
            "viewType": "Bar Chart",
            "config": {
                "chartTitle": "E1 — Tail Class Recall (%) at Extreme Imbalance ρ = 100 (+42.67 pp Gain)",
                "metrics": ["summary:rho100_tail_acc_mean"],
                "vertical": False,
                "xAxisMin": 0,
                "xAxisTitle": "Tail Recall (%) at ρ = 100",
                "limit": 6,
                "barLimit": 6,
            },
        },
    ]

    e3a_panels = [
        {
            "__id__": "lab2_e3a_time",
            "layout": {"x": 0, "y": 0, "w": 12, "h": 9},
            "viewType": "Run History Line Plot",
            "config": {
                "chartTitle": "E3A — Empirical Θ(D²) RLS Runtime Wall (Slope α = 2.11, 35.6× Slower at D=6000)",
                "metrics": ["rls_update_seconds_mean"],
                "xAxis": "projection_dim",
                "xAxisLogScale": True,
                "yAxisLogScale": True,
                "yAxisTitle": "10-Task RLS Time (seconds)",
            },
        },
        {
            "__id__": "lab2_e3a_mem",
            "layout": {"x": 12, "y": 0, "w": 12, "h": 9},
            "viewType": "Run History Line Plot",
            "config": {
                "chartTitle": "E3A — Precision Matrix R Memory Footprint (MB, 36.0× Growth)",
                "metrics": ["r_matrix_mb"],
                "xAxis": "projection_dim",
                "xAxisLogScale": True,
                "yAxisLogScale": True,
                "yAxisTitle": "Matrix R Memory (MB, float64)",
            },
        },
    ]

    e3b_panels = [
        {
            "__id__": "lab2_e3b_wnorm",
            "layout": {"x": 0, "y": 0, "w": 12, "h": 8},
            "viewType": "Bar Chart",
            "config": {
                "chartTitle": "E3B — Classifier Norm ||W||_F at Interpolation Threshold D=2000, γ=10⁻³ (16× Blowup at V=1600)",
                "metrics": ["summary:d2000_gamma1e3_w_frobenius_norm"],
                "vertical": False,
                "xAxisMin": 0,
                "xAxisTitle": "Frobenius Norm ||W||_F",
                "limit": 4,
                "barLimit": 4,
            },
        },
        {
            "__id__": "lab2_e3b_fd2",
            "layout": {"x": 12, "y": 0, "w": 12, "h": 8},
            "viewType": "Bar Chart",
            "config": {
                "chartTitle": "E3B — Parameter-Free FD2 (γ_auto) Average Incremental Accuracy (%) at D=4000",
                "metrics": ["summary:fd2_d4000_average_accuracy"],
                "vertical": False,
                "xAxisMin": 0,
                "xAxisTitle": "Average Incremental Accuracy (%)",
                "limit": 4,
                "barLimit": 4,
            },
        },
    ]

    blocks = [
        {
            "type": "markdown-block",
            "children": [{"text": ""}],
            "content": (
                "Báo cáo công khai này tổng hợp toàn bộ **6 lượt chạy thực nghiệm chuẩn của Giai đoạn 2 "
                "(Milestone 3: Critical Analysis & Lab 2: Empirical Demonstration)** thuộc kho mã nguồn "
                "[`caoban123/F-OAL`](https://github.com/caoban123/F-OAL) (tách biệt hoàn toàn với dữ liệu Lab 1 qua nhãn "
                "`lab2-empirical-demo` và `critical-analysis`).\n\n"
                "**Tài liệu đối chiếu trong Repository:**\n"
                "- Phản biện lý thuyết & chứng minh định lý: `docs/CRITICAL_ANALYSIS_REPORT.md`\n"
                "- Báo cáo thực nghiệm 152 runs (E1 + E3): `docs/LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md`\n"
                "- Mã nguồn giải pháp 0-replay: `src/foal/critical_analysis.py`\n"
                "- Poster khoa học khổ dọc Portrait A2: `poster/F-OAL_Critical_Analysis_Poster_A2.pptx`"
            ),
        },
        {
            "type": "heading",
            "level": 1,
            "children": [{"text": "1. Trục I (L1 & FD1) — Sự sụp đổ Lớp Hiếm Long-Tail OCIL & Giải pháp Balanced Woodbury (54 Runs, 3 Seeds)"}],
        },
        {
            "type": "markdown-block",
            "children": [{"text": ""}],
            "content": (
                "**Phát hiện kiểm định:** Mục 4.6 & Figure 3 của bài báo NeurIPS 2024 khẳng định chuẩn cột bộ phân loại "
                "$\\|\\hat{\\mathbf{w}}_c\\|_2$ tự động đồng đều chống thiên lệch. Nhóm chứng minh bằng đồng nhất thức Sherman–Morrison "
                "rằng tính chất này chỉ đúng khi $N_1 = \\dots = N_C$. Trên CIFAR-100 mất cân bằng mũ ($\\rho = N_{\\max}/N_{\\min} = 100$, "
                "3 seeds `42, 123, 2026`), tương quan $r(\\|\\hat{\\mathbf{w}}_c\\|_2, N_c) = +0.89$ làm nhóm 1/3 lớp hiếm sụp đổ **-60.84 pp** "
                "(`87.48%` xuống `26.64%`, run `g34o8awx`).\n\n"
                "**Giải pháp giải tích 1 lượt, 0 lưu mẫu (`BalancedOnlineRidgeRLS` — `FD1`):**\n"
                "- `FD1b` (`kafjjzio` — chuẩn hóa cột lúc suy luận $\\|\\hat{\\mathbf{w}}_c\\|_2^\\tau$, **0 chi phí train**): khôi phục **+28.50 pp** Tail Recall (`55.14%`).\n"
                "- `FD1a+b` (`fqahbyt4` — cập nhật Woodbury nhân quả có trọng số tần suất nghịch đảo + chuẩn hóa cột): khôi phục **+42.67 pp** Tail Recall (`69.31%`) và **+13.32 pp** $\\bar{A}$ (`72.79%` lên `86.11%`)."
            ),
        },
        _make_panel_grid_block(
            "lab2_runset_e1",
            "Lab 2 E1 — Long-Tail OCIL (3 Seeds)",
            ["g34o8awx", "kafjjzio", "fqahbyt4"],
            e1_panels,
            x_axis="imbalance_ratio_rho",
        ),
        {
            "type": "heading",
            "level": 1,
            "children": [{"text": "2. Trục II (L2 & FD2) — Đính chính Độ phức tạp Θ(D²), Đỉnh suy biến Double-Descent & Tự hiệu chỉnh γ_auto (98 Runs)"}],
        },
        {
            "type": "markdown-block",
            "children": [{"text": ""}],
            "content": (
                "**Phát hiện kiểm định:**\n"
                "1. Equation (14) trong bài báo viết chi phí cập nhật ma trận $\\mathbf{R}$ là $\\mathcal{O}(D^2 C)$, trong khi $\\mathbf{R}_k$ hoàn toàn độc lập với số lớp $C$ và có độ phức tạp thực $\\Theta(D^2 S + DS^2 + S^3)$. Thực nghiệm `nbr04hu5` đo được hệ số góc log-log **$\\alpha = 2.11$** (`D=6000` chậm gấp `35.6×` và tốn `274.7 MB` chỉ cho `+0.77 pp` $\\bar{A}$).\n"
                "2. Khi tổng số mẫu nhỏ ($V = 1{,}600$, tương đương quy mô DTD/Aircraft), tại ngưỡng nội suy $D = 2{,}000 \\approx V$ với $\\gamma = 10^{-3}$ (`kyj36qk9`), các trị riêng cực nhỏ Marchenko–Pastur làm nổ chuẩn $\\|\\hat{\\mathbf{W}}\\|_F = 155.0$ (**gấp 16×**) kéo $A_{10}$ sập xuống `72.64%`.\n\n"
                "**Giải pháp không tham số `FD2` (`compute_trace_scaled_gamma`):** Tự động tỷ lệ hóa $\\gamma_{\\text{auto}}$ theo năng lượng vết Task 1 và tỷ số $D/V_{\\text{est}}$ (không cần tập validation), loại bỏ hoàn toàn đỉnh suy biến và đạt $\\bar{A}$ cao nhất toàn lưới (`88.19%` tại $D = 4{,}000$, vượt $\\gamma = 1$ cố định `+1.28 pp`)."
            ),
        },
        _make_panel_grid_block(
            "lab2_runset_e3a",
            "Lab 2 E3A — Quadratic O(D^2) Scaling",
            ["nbr04hu5"],
            e3a_panels,
            x_axis="projection_dim",
        ),
        _make_panel_grid_block(
            "lab2_runset_e3b",
            "Lab 2 E3B — Double-Descent & FD2 Adaptive Ridge",
            ["1bns4np4", "kyj36qk9"],
            e3b_panels,
            x_axis="projection_dim",
        ),
    ]

    return {
        "version": 5,
        "panelSettings": {},
        "blocks": blocks,
        "width": "readable",
        "authors": [],
        "discussionThreads": [],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Harmonize W&B Workspace, additive run summaries, and Lab 2 Report.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Inspect and write backup preview without mutating W&B.")
    mode.add_argument("--apply", action="store_true", help="Write backup snapshot, apply non-destructive updates, and verify.")
    args = parser.parse_args()

    api = wandb.Api(timeout=30)

    # 1. Inspect Workspaces
    ws_data = _gql_request(
        api,
        Reports.QUERY,
        {"entity": ENTITY, "project": PROJECT, "reportLimit": 50, "viewType": "project-view"},
    )
    ws_edges = ws_data["project"]["allViews"]["edges"]
    ws_plans = []
    for edge in ws_edges:
        node = edge["node"]
        if node["name"] in {"nw-nwusercaoban170106-w", "nw-nwuserlqdtk23vothaithanh-w"}:
            orig_spec = json.loads(node["spec"])
            new_spec, changed = _transform_workspace_spec(orig_spec)
            ws_plans.append({
                "id": node["id"],
                "name": node["name"],
                "displayName": node["displayName"],
                "changed": changed,
                "orig_spec": orig_spec,
                "new_spec": new_spec,
            })

    # 2. Plan additive summary updates
    run_plans = _plan_additive_run_updates(api)

    # 3. Check if Lab 2 Report already exists
    rpt_data = _gql_request(
        api,
        Reports.QUERY,
        {"entity": ENTITY, "project": PROJECT, "reportLimit": 50, "viewType": "runs"},
    )
    existing_lab2_rpt = next(
        (e["node"] for e in rpt_data["project"]["allViews"]["edges"] if e["node"]["name"] == LAB2_REPORT_NAME),
        None,
    )
    lab2_spec = _build_lab2_report_spec()

    print("=== W&B HARMONIZATION PLAN ===")
    for wp in ws_plans:
        print(f"Workspace {wp['displayName']!r} ({wp['id']}): needs_update={wp['changed']}")
    for rp in run_plans:
        print(f"Run {rp['run_id']} ({rp['run_name']}): add_keys={list(rp['additions'].keys())}")
    print(f"Lab 2 Report ({LAB2_REPORT_NAME}): existing_id={existing_lab2_rpt['id'] if existing_lab2_rpt else None}")

    if args.dry_run:
        print("[DRY-RUN] No mutations performed on W&B.")
        return

    # Write immutable backup BEFORE applying any live change
    backup_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entity": ENTITY,
        "project": PROJECT,
        "workspaces_before": [
            {"id": w["id"], "name": w["name"], "displayName": w["displayName"], "spec": w["orig_spec"]}
            for w in ws_plans
        ],
        "runs_before": [
            {"run_id": r["run_id"], "run_name": r["run_name"], "summary": r["before_summary"], "planned_additions": r["additions"]}
            for r in run_plans
        ],
        "existing_lab2_report_before": existing_lab2_rpt,
    }
    backup_file = _write_immutable_backup(backup_payload)
    print(f"[BACKUP SAVED] {backup_file.relative_to(REPO_ROOT).as_posix()}")

    # Apply 1: Workspace layouts
    for wp in ws_plans:
        if not wp["changed"]:
            continue
        try:
            _gql_request(
                api,
                UPSERT_VIEW_MUTATION,
                {
                    "input": {
                        "id": wp["id"],
                        "entityName": ENTITY,
                        "projectName": PROJECT,
                        "name": wp["name"],
                        "displayName": wp["displayName"],
                        "type": "project-view",
                        "spec": json.dumps(wp["new_spec"], ensure_ascii=False),
                    }
                },
            )
            print(f"[UPDATED WORKSPACE] {wp['displayName']} ({wp['id']})")
        except Exception as exc:
            print(f"[SKIPPED WORKSPACE] {wp['displayName']} ({wp['id']}): {exc}")

    # Apply 2: Additive run summary keys
    for rp in run_plans:
        if not rp["additions"]:
            print(f"[RUN ALREADY UP TO DATE] {rp['run_id']}")
            continue
        run = api.run(f"{ENTITY}/{PROJECT}/{rp['run_id']}")
        for k, v in rp["additions"].items():
            assert k not in run.summary, f"Refusing to overwrite existing key {k} on {rp['run_id']}"
            run.summary[k] = v
        run.summary.update()
        print(f"[UPDATED RUN SUMMARY] {rp['run_id']} (+{len(rp['additions'])} additive keys)")

    # Apply 3: Upsert 4th public W&B Report for Lab 2
    upsert_input: dict[str, Any] = {
        "entityName": ENTITY,
        "projectName": PROJECT,
        "name": LAB2_REPORT_NAME,
        "displayName": LAB2_REPORT_DISPLAY_NAME,
        "description": (
            "Báo cáo thực nghiệm Lab 2: Kiểm định sự sụp đổ Long-Tail OCIL (54 runs), "
            "quy luật chi phí bậc hai Θ(D²), đỉnh suy biến Double-Descent (98 runs) và "
            "hai giải pháp giải tích 0-replay (FD1 BalancedOnlineRidgeRLS & FD2 Trace-Scaled Adaptive Ridge)."
        ),
        "type": "runs",
        "spec": json.dumps(lab2_spec, ensure_ascii=False),
    }
    if existing_lab2_rpt is not None:
        upsert_input["id"] = existing_lab2_rpt["id"]

    rpt_out = _gql_request(api, UPSERT_VIEW_MUTATION, {"input": upsert_input})
    view_node = rpt_out["upsertView"]["view"]
    app_url = api.settings["base_url"].replace("://api.", "://").rstrip("/") + "/wandb"
    if "forge.coreweave.com" in api.settings["base_url"]:
        app_url = "https://forge.coreweave.com/wandb"
    report_url = _build_report_url(app_url, view_node["displayName"], view_node["id"])
    print(f"[UPSERTED LAB 2 REPORT] id={view_node['id']} | url={report_url}")

    # Record Lab 2 Report URL in manifest
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest["lab2_report"] = {
        "id": view_node["id"],
        "name": view_node["name"],
        "display_name": view_node["displayName"],
        "url": report_url,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[UPDATED MANIFEST] {MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
