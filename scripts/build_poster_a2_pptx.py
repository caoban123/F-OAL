# -*- coding: utf-8 -*-
"""Build the single-slide Portrait A2 Academic Research Poster (.pptx) for F-OAL.

Strictly enforces all HCMUS Term Project (`Empirical Demonstration`) & NeurIPS/ICML Poster rules:
- Single `.pptx` file containing EXACTLY ONE slide in Portrait A2 (`420.0 mm x 594.0 mm`).
- 60-30-10 colorblind-safe academic palette (`#F1F5F9` soft canvas, `#FFFFFF` cards, `#0F2942`
  Oxford Navy structure, `#C2410C` Vermilion critique accents, `#0F766E` Teal remedy accents).
- Wired side-by-side layout: Theoretical Critique + Analytic Remedy (left) directly paired with
  Measured Empirical Proof + Live W&B Run IDs (right) for the 2 Primary Axes, plus 3 Evidenced
  Strengths, Pipeline Audit Schematic, Synthesis Table (caption above), and IEEE References.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Mm, Pt

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "poster" / "assets"
OUT_PPTX = REPO_ROOT / "poster" / "F-OAL_Critical_Analysis_Poster_A2.pptx"
OUT_PREVIEW = REPO_ROOT / "poster" / "F-OAL_Critical_Analysis_Poster_A2_preview.png"

# Exact ISO A2 Portrait Dimensions
A2_WIDTH_MM = 420.0
A2_HEIGHT_MM = 594.0

# Palette
C_CANVAS = RGBColor(0xF1, 0xF5, 0xF9)     # #F1F5F9 soft glare-free cool gray
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_NAVY = RGBColor(0x0F, 0x29, 0x42)       # #0F2942 institutional Oxford navy
C_TEAL = RGBColor(0x0F, 0x76, 0x6E)       # #0F766E deep academic teal
C_TEAL_BRIGHT = RGBColor(0x2D, 0xD4, 0xBF)
C_TEAL_TINT = RGBColor(0xF0, 0xFD, 0xFA)
C_VERM = RGBColor(0x9A, 0x34, 0x12)       # #9A3412 deep burnt vermilion
C_VERM_TINT = RGBColor(0xFF, 0xF7, 0xED)
C_SAPPHIRE = RGBColor(0x03, 0x69, 0xA1)
C_ICE = RGBColor(0xF8, 0xFA, 0xFC)
C_BORDER = RGBColor(0xCB, 0xD5, 0xE1)
C_TEXT = RGBColor(0x1E, 0x29, 0x3B)
C_MUTED = RGBColor(0x47, 0x55, 0x69)
C_LIGHT_TEXT = RGBColor(0xE2, 0xE8, 0xF0)

FONT_MAIN = "Segoe UI"


def add_rect(
    slide,
    x_mm: float,
    y_mm: float,
    w_mm: float,
    h_mm: float,
    *,
    fill: RGBColor = C_WHITE,
    border: RGBColor | None = C_BORDER,
    border_pt: float = 1.0,
    rounded: bool = True,
):
    """Add a crisp rectangle or rounded card shape in millimeter coordinates."""
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Mm(x_mm), Mm(y_mm), Mm(w_mm), Mm(h_mm))
    if rounded and len(shp.adjustments) > 0:
        shp.adjustments[0] = 0.02
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if border is not None:
        shp.line.color.rgb = border
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_section_header(
    slide,
    x_mm: float,
    y_mm: float,
    w_mm: float,
    badge: str,
    title: str,
    *,
    bg: RGBColor = C_NAVY,
    h_mm: float = 9.5,
) -> None:
    """Add a structured conference-style section banner at the top of a card."""
    bar = add_rect(slide, x_mm, y_mm, w_mm, h_mm, fill=bg, border=None, rounded=False)
    tf = bar.text_frame
    tf.word_wrap = True
    tf.margin_left = Mm(3.2)
    tf.margin_right = Mm(2.5)
    tf.margin_top = Mm(0.8)
    tf.margin_bottom = Mm(0.8)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r_b = p.add_run()
    r_b.text = f"{badge}  |  "
    r_b.font.name = FONT_MAIN
    r_b.font.size = Pt(12.5)
    r_b.font.bold = True
    r_b.font.color.rgb = C_TEAL_BRIGHT if bg == C_NAVY else RGBColor(0xFD, 0xE6, 0x8A)

    r_t = p.add_run()
    r_t.text = title
    r_t.font.name = FONT_MAIN
    r_t.font.size = Pt(13.5)
    r_t.font.bold = True
    r_t.font.color.rgb = C_WHITE


def add_rich_bullets(
    slide,
    x_mm: float,
    y_mm: float,
    w_mm: float,
    h_mm: float,
    items: Sequence[tuple[str, str]],
    *,
    font_pt: float = 13.0,
    lead_color: RGBColor = C_NAVY,
    body_color: RGBColor = C_TEXT,
    space_after_pt: float = 5.0,
) -> None:
    """Add left-aligned ragged-right bullet items with bold colored lead-in phrases."""
    tb = slide.shapes.add_textbox(Mm(x_mm), Mm(y_mm), Mm(w_mm), Mm(h_mm))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Mm(1.0)
    tf.margin_right = Mm(1.0)
    tf.margin_top = Mm(0.5)
    tf.margin_bottom = Mm(0.5)

    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after_pt)
        p.line_spacing = 1.15

        r1 = p.add_run()
        r1.text = f"▸ {lead}: "
        r1.font.name = FONT_MAIN
        r1.font.size = Pt(font_pt)
        r1.font.bold = True
        r1.font.color.rgb = lead_color

        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_MAIN
        r2.font.size = Pt(font_pt)
        r2.font.bold = False
        r2.font.color.rgb = body_color


def fit_picture_in_box(
    slide,
    img_path: Path,
    box_x_mm: float,
    box_y_mm: float,
    box_w_mm: float,
    box_h_mm: float,
) -> None:
    """Insert a high-DPI PNG centered inside a bounding box without aspect-ratio distortion."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    img_ratio = iw / float(ih)
    box_ratio = box_w_mm / float(box_h_mm)
    if img_ratio > box_ratio:
        w_mm = box_w_mm
        h_mm = box_w_mm / img_ratio
    else:
        h_mm = box_h_mm
        w_mm = box_h_mm * img_ratio
    x_mm = box_x_mm + (box_w_mm - w_mm) / 2.0
    y_mm = box_y_mm + (box_h_mm - h_mm) / 2.0
    slide.shapes.add_picture(str(img_path), Mm(x_mm), Mm(y_mm), width=Mm(w_mm), height=Mm(h_mm))


def add_caption(
    slide,
    x_mm: float,
    y_mm: float,
    w_mm: float,
    h_mm: float,
    label: str,
    text: str,
    *,
    font_pt: float = 11.2,
    bold_color: RGBColor = C_NAVY,
) -> None:
    """Add a NeurIPS-style figure/table caption."""
    tb = slide.shapes.add_textbox(Mm(x_mm), Mm(y_mm), Mm(w_mm), Mm(h_mm))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Mm(1.0)
    tf.margin_right = Mm(1.0)
    tf.margin_top = Mm(0.2)
    tf.margin_bottom = Mm(0.2)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.line_spacing = 1.10

    r1 = p.add_run()
    r1.text = f"{label} "
    r1.font.name = FONT_MAIN
    r1.font.size = Pt(font_pt)
    r1.font.bold = True
    r1.font.color.rgb = bold_color

    r2 = p.add_run()
    r2.text = text
    r2.font.name = FONT_MAIN
    r2.font.size = Pt(font_pt)
    r2.font.color.rgb = C_MUTED


def build_poster() -> Path:
    """Construct the complete Portrait A2 `.pptx` poster."""
    prs = Presentation()
    prs.slide_width = Mm(A2_WIDTH_MM)
    prs.slide_height = Mm(A2_HEIGHT_MM)
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # 0. Soft Glare-Free Off-White Canvas (60-30-10 Rule)
    add_rect(slide, 0, 0, A2_WIDTH_MM, A2_HEIGHT_MM, fill=C_CANVAS, border=None, rounded=False)

    # =========================================================================
    # BAND 0: FULL-WIDTH ACADEMIC HEADER (y = 0 .. 58 mm)
    # =========================================================================
    add_rect(slide, 0, 0, A2_WIDTH_MM, 58.0, fill=C_NAVY, border=None, rounded=False)
    add_rect(slide, 0, 0, A2_WIDTH_MM, 3.0, fill=C_TEAL_BRIGHT, border=None, rounded=False)

    tb_hdr = slide.shapes.add_textbox(Mm(11.0), Mm(4.5), Mm(322.0), Mm(51.5))
    tf_h = tb_hdr.text_frame
    tf_h.word_wrap = True
    tf_h.margin_left = tf_h.margin_top = tf_h.margin_right = tf_h.margin_bottom = Mm(0.5)

    p_kick = tf_h.paragraphs[0]
    p_kick.space_after = Pt(2.5)
    rk = p_kick.add_run()
    rk.text = "TRƯỜNG ĐHKHTN, ĐHQG-HCM (HCMUS)  ·  NHẬP MÔN HỌC MÁY  ·  CRITICAL ANALYSIS & EMPIRICAL DEMONSTRATION"
    rk.font.name = FONT_MAIN
    rk.font.size = Pt(11.5)
    rk.font.bold = True
    rk.font.color.rgb = C_TEAL_BRIGHT

    p_title = tf_h.add_paragraph()
    p_title.space_after = Pt(3.0)
    p_title.line_spacing = 1.08
    rt = p_title.add_run()
    rt.text = "Beyond Balanced Benchmarks: Critiquing F-OAL's Squared-Error Bias & Spectral Scaling Limits"
    rt.font.name = FONT_MAIN
    rt.font.size = Pt(25.0)
    rt.font.bold = True
    rt.font.color.rgb = C_WHITE

    p_sub = tf_h.add_paragraph()
    p_sub.space_after = Pt(2.5)
    rs = p_sub.add_run()
    rs.text = "Phân tích phản biện lý thuyết & Giải pháp giải tích không lưu mẫu (Zero-Replay) cho F-OAL (Zhuang et al., NeurIPS 2024 [1])"
    rs.font.name = FONT_MAIN
    rs.font.size = Pt(13.2)
    rs.font.italic = True
    rs.font.color.rgb = C_LIGHT_TEXT

    p_auth = tf_h.add_paragraph()
    ra = p_auth.add_run()
    ra.text = "Nhóm thực hiện: 24122014 · 24122029 (Nguyễn Cao Bản) · 24122050   |   Repo: github.com/caoban123/F-OAL   |   W&B Tag: lab2-empirical-demo"
    ra.font.name = FONT_MAIN
    ra.font.size = Pt(11.8)
    ra.font.bold = True
    ra.font.color.rgb = RGBColor(0xFD, 0xE6, 0x8A)

    # Top-right QR Badges (GitHub & W&B)
    for idx, (qr_file, qr_lbl) in enumerate([
        ("qr_github.png", "GitHub Repo"),
        ("qr_wandb.png", "W&B Runs"),
    ]):
        qx = 337.0 + idx * 38.0
        add_rect(slide, qx, 6.5, 34.5, 45.5, fill=C_WHITE, border=C_TEAL_BRIGHT, border_pt=1.2, rounded=True)
        fit_picture_in_box(slide, ASSET_DIR / qr_file, qx + 2.2, 8.0, 30.1, 30.1)
        tb_q = slide.shapes.add_textbox(Mm(qx), Mm(38.5), Mm(34.5), Mm(12.5))
        tf_q = tb_q.text_frame
        tf_q.margin_left = tf_q.margin_right = tf_q.margin_top = tf_q.margin_bottom = Mm(0.2)
        pq = tf_q.paragraphs[0]
        pq.alignment = PP_ALIGN.CENTER
        rq = pq.add_run()
        rq.text = qr_lbl
        rq.font.name = FONT_MAIN
        rq.font.size = Pt(9.8)
        rq.font.bold = True
        rq.font.color.rgb = C_NAVY

    # =========================================================================
    # BAND 1: VISUAL HOOK — PIPELINE AUDIT + 3 EVIDENCED STRENGTHS (y = 63 .. 157 mm)
    # =========================================================================
    # Left Card: Redrawn F-OAL Architecture & Fault-Line Audit (x = 8 .. 216, w = 208, h = 94)
    add_rect(slide, 8.0, 63.0, 208.0, 94.0, fill=C_WHITE, border=C_BORDER, border_pt=1.1)
    add_section_header(slide, 8.0, 63.0, 208.0, "01", "KIẾN TRÚC F-OAL & BẢN ĐỒ ĐIỂM GÃY LÝ THUYẾT", bg=C_NAVY)
    fit_picture_in_box(slide, ASSET_DIR / "asset_pipeline_critique.png", 10.5, 74.0, 203.0, 71.0)
    add_caption(
        slide, 10.5, 145.5, 203.0, 10.5,
        "Hình 1 (Tự vẽ lại từ [1]):",
        "Luồng học một lượt F-OAL và 3 vị trí xuất hiện giả định ẩn được kiểm định trong nghiên cứu này.",
        font_pt=10.8,
    )

    # Right Card: 3 Evidenced Strengths S1, S2, S3 (x = 221 .. 412, w = 191, h = 94)
    add_rect(slide, 221.0, 63.0, 191.0, 94.0, fill=C_WHITE, border=C_BORDER, border_pt=1.1)
    add_section_header(slide, 221.0, 63.0, 191.0, "02", "BA ĐIỂM MẠNH ĐƯỢC BẢO CHỨNG (S1–S3)", bg=C_TEAL)

    strengths = [
        (
            "S1 · Nghiệm đóng 1 lượt = Học đồng thời (Eq. 10–13 [1])",
            "Đẳng thức Woodbury khớp nghiệm Ridge toàn cục (< 10⁻¹² ở float64), triệt tiêu gradient recency bias trên luồng cân bằng: CIFAR-100 đạt Ā = 90.40%, F_past = 5.16% [3v53tmq5] vượt xa ER (86.67%, F = 16.74% [xlcme1c3]) & LwF-AdamW (F = 94.99% [ifl3rz2u]).",
        ),
        (
            "S2 · Bộ nhớ hằng số O(D² + DC) & Không lưu ảnh cũ",
            "Đóng băng 85.8M tham số ViT giúp bỏ hoàn toàn bước backward (~70% FLOPs [2]). Tại D = 1,000, ma trận R chỉ chiếm 7.63 MB; toàn bộ cập nhật RLS 10 tasks trên DTD chỉ tốn 0.79–1.22s (Ā = 85.85 ± 0.92% [nyukb6j0]).",
        ),
        (
            "S3 · Cộng hưởng FF + Smooth Projection ở miền tinh (Cover [3])",
            "Tái lập Table 4 (3 seeds: 42, 123, 2026) chứng minh mở rộng phi tuyến σ(hP) đặc biệt quyết định khi lệch miền/phân loại tinh: đóng góp gộp +10.61 ± 0.50 pp trên DTD và +6.46 ± 0.43 pp trên FGVC-Aircraft (64.22 ± 3.70% [2qksp3kv]).",
        ),
    ]
    add_rich_bullets(
        slide, 224.0, 74.0, 185.0, 81.5, strengths,
        font_pt=11.6, lead_color=C_TEAL, space_after_pt=4.2,
    )

    # =========================================================================
    # BAND 2: PRIMARY AXIS I — LONG-TAIL CLASS-IMBALANCE COLLAPSE (y = 162 .. 349 mm)
    # =========================================================================
    # Left Theory + FD1 Card (x = 8 .. 160, w = 152, h = 187)
    add_rect(slide, 8.0, 162.0, 152.0, 187.0, fill=C_WHITE, border=C_VERM, border_pt=1.3)
    add_section_header(slide, 8.0, 162.0, 152.0, "03A", "TRỤC I — PHẢN BIỆN LÝ THUYẾT L1 & GIẢI PHÁP FD1", bg=C_VERM)

    fit_picture_in_box(slide, ASSET_DIR / "asset_eq_axis1.png", 10.5, 173.0, 147.0, 55.0)

    axis1_bullets = [
        (
            "Giả định cân bằng ẩn sau Figure 3 [1]",
            "Bài báo khẳng định chuẩn cột ||w_c||₂ đồng đều chống thiên lệch (Mục 4.6). Biến đổi Sherman–Morrison chứng minh điều này chỉ đúng khi N₁ = ... = N_C! Với nhãn {0, 1}, lớp hiếm c bị V − N_c mẫu âm ép y_{i,c} = 0 trên nón kích hoạt dương σ(hP) ∈ (0,1)^D.",
        ),
        (
            "Hệ quả lý thuyết khi ρ = N_max / N_min ≫ 1",
            "Biên độ ||w_t||₂ của lớp đuôi suy giảm theo O(1/ρ), làm điểm dự đoán φ(x)w_t tụt dưới lớp giàu mẫu ngay cả khi không quên đặc trưng.",
        ),
        (
            "Giải pháp FD1 (BalancedOnlineRidgeRLS)",
            "Nhân trọng số tần suất nghịch đảo nhân quả ω_i = (N_ref / N̂_{y_i})^\u03b2 ngay trong cập nhật Woodbury kết hợp chuẩn hóa cột suy luận ||w_c||₂^\u03c4 (0 mẫu lưu trữ, 1 lượt duyệt).",
        ),
    ]
    add_rich_bullets(
        slide, 11.0, 229.5, 146.0, 118.0, axis1_bullets,
        font_pt=11.9, lead_color=C_VERM, space_after_pt=4.5,
    )

    # Right Empirical E1 Card (x = 165 .. 412, w = 247, h = 187)
    add_rect(slide, 165.0, 162.0, 247.0, 187.0, fill=C_WHITE, border=C_NAVY, border_pt=1.3)
    add_section_header(slide, 165.0, 162.0, 247.0, "03B", "CHỨNG MINH THỰC NGHIỆM E1: 54 RUNS LONG-TAIL OCIL (3 SEEDS)", bg=C_NAVY)

    fit_picture_in_box(slide, ASSET_DIR / "asset_fig_axis1.png", 168.0, 173.0, 241.0, 156.5)
    add_caption(
        slide, 168.0, 330.2, 241.0, 17.5,
        "Hình 2 (CIFAR-100, 60,000 ảnh, seeds 42/123/2026):",
        "Gán hạn ngạch mũ ngẫu nhiên đều khắp 10 tasks để tách khỏi hiệu ứng thứ tự task. Tại ρ = 100, F-OAL gốc mất −60.84 pp lớp đuôi (87.48% → 26.64% [g34o8awx]); FD1 khôi phục +42.67 pp Tail Recall (69.31%) & +13.32 pp Ā (86.11% [fqahbyt4], Norm-only [kafjjzio]).",
        font_pt=10.6,
    )

    # =========================================================================
    # BAND 3: PRIMARY AXIS II — COMPLEXITY BUG, O(D^2) & DOUBLE-DESCENT (y = 354 .. 536 mm)
    # =========================================================================
    # Left Theory + FD2 Card (x = 8 .. 160, w = 152, h = 182)
    add_rect(slide, 8.0, 354.0, 152.0, 182.0, fill=C_WHITE, border=C_VERM, border_pt=1.3)
    add_section_header(slide, 8.0, 354.0, 152.0, "04A", "TRỤC II — PHẢN BIỆN LÝ THUYẾT L2 & GIẢI PHÁP FD2", bg=C_VERM)

    fit_picture_in_box(slide, ASSET_DIR / "asset_eq_axis2.png", 10.5, 365.0, 147.0, 54.5)

    axis2_bullets = [
        (
            "Đính chính lỗi đại số Eq. (1) & Eq. (14) [1]",
            "Eq. (14) viết cập nhật R tốn max{O(SDC), O(D²C)} là sai biến: Eq. (11)/(13) hoàn toàn độc lập với C, chi phí thực là Θ(D²S + DS² + S³)/batch ⇒ nổ bậc hai Θ(VD²) toàn khóa (D = 16,000 tốn > 2 GB chỉ riêng ma trận R).",
        ),
        (
            "Bẫy suy biến phổ D ≈ V (Giải mã Table 6 [1])",
            "Khi cỡ mẫu nhỏ (DTD V = 1,600; Aircraft V = 3,334 < D), rank(ΦᵀΦ) ≤ V < D sinh ra các trị riêng sát 0. Nếu γ ≪ 1, bộ lọc Tikhonov khuếch đại nhiễu 1/(2√γ) làm nổ ||Ŵ||_F (giải thích Aircraft rơi 61.1% → 22.4% ở Table 6).",
        ),
        (
            "Giải pháp FD2 (Trace-Scaled Adaptive γ_auto)",
            "Định cỡ tự động γ_auto theo năng lượng vết Task 1 và tỷ số quá tham số hóa D/V_est ngay từ đầu luồng mà không cần tách tập validation.",
        ),
    ]
    add_rich_bullets(
        slide, 11.0, 421.0, 146.0, 113.5, axis2_bullets,
        font_pt=11.7, lead_color=C_VERM, space_after_pt=4.0,
    )

    # Right Empirical E3 Card (x = 165 .. 412, w = 247, h = 182)
    add_rect(slide, 165.0, 354.0, 247.0, 182.0, fill=C_WHITE, border=C_NAVY, border_pt=1.3)
    add_section_header(slide, 165.0, 354.0, 247.0, "04B", "CHỨNG MINH THỰC NGHIỆM E3: QUY LUẬT Θ(D²) & DOUBLE-DESCENT", bg=C_NAVY)

    fit_picture_in_box(slide, ASSET_DIR / "asset_fig_axis2.png", 168.0, 365.0, 241.0, 151.5)
    add_caption(
        slide, 168.0, 517.2, 241.0, 17.5,
        "Hình 3 (98 runs quét D = 250..6000 × γ = 10⁻³..10²):",
        "Độ dốc thực nghiệm α = 2.11 khớp sát Θ(D²) [nbr04hu5]. Ở chế độ ít mẫu V = 1,600 [kyj36qk9], đúng tại ngưỡng nội suy D = 2,000 ≈ V (γ = 10⁻³), chuẩn ||Ŵ||_F nổ gấp 16× (155.0) kéo A₁₀ sập xuống 72.64%; công thức không tham số FD2 đạt Ā cao nhất toàn lưới (88.19% ở D = 4,000).",
        font_pt=10.6,
    )

    # =========================================================================
    # BAND 4: FOOTER — SYNTHESIS TABLE (CAPTION ABOVE) & IEEE REFS (y = 541 .. 588 mm)
    # =========================================================================
    # Left Card: Synthesis Table (x = 8 .. 282, w = 274, h = 47)
    add_rect(slide, 8.0, 541.0, 274.0, 47.0, fill=C_WHITE, border=C_BORDER, border_pt=1.1)
    add_caption(
        slide, 10.5, 542.0, 269.0, 6.2,
        "Bảng 1 — Đối chiếu Tuyên bố Bài báo, Phát hiện Kiểm định & Giải pháp:",
        "(Điểm yếu L3 xem chi tiết tại README.md §16).",
        font_pt=10.1,
    )

    rows_data = [
        [" Tuyên bố trong Bài báo [1]", " Phát hiện Phản biện & Đo đạc", " Giải pháp Giải tích 0-Replay", " Hiệu quả Kiểm chứng"],
        [" Chuẩn ‖w_c‖₂ đồng đều (Fig. 3)", " Sụp đổ O(1/ρ) khi lệch lớp (r = +0.89)", " Balanced Woodbury + Norm (FD1)", " +42.7 pp Tail, +13.3 pp Ā [fqahbyt4]"],
        [" R tốn O(D²C); γ = 1 cố định", " Đúng là Θ(D²S); nổ ‖Ŵ‖_F khi D ≈ V", " Trace-Adaptive Ridge γ_auto (FD2)", " 88.2% Ā (+9.4 pp vs γ=10⁻³) [kyj36qk9]"],
        [" Chiếu ngẫu nhiên N(0, 1) (tr. 4)", " Bão hòa >81% Sigmoid (README §16)", " Luật phương sai σ_P² = 1/(3d_in)", " +16.5 pp DTD, +18.8 pp Aircraft [C1]"],
    ]
    tbl_shape = slide.shapes.add_table(4, 4, Mm(10.5), Mm(549.5), Mm(269.0), Mm(36.5))
    tbl = tbl_shape.table
    col_widths = [59.0, 74.0, 68.0, 68.0]
    for c_i, cw in enumerate(col_widths):
        tbl.columns[c_i].width = Mm(cw)

    for r_i, row in enumerate(rows_data):
        for c_i, val in enumerate(row):
            cell = tbl.cell(r_i, c_i)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Mm(1.2)
            cell.margin_right = Mm(1.0)
            cell.margin_top = Mm(0.4)
            cell.margin_bottom = Mm(0.4)
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = C_NAVY
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = C_ICE
            else:
                cell.fill.fore_color.rgb = C_WHITE

            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            run.text = val.strip()
            run.font.name = FONT_MAIN
            run.font.size = Pt(9.8 if r_i > 0 else 10.0)
            run.font.bold = (r_i == 0 or c_i == 3)
            if r_i == 0:
                run.font.color.rgb = C_WHITE
            elif c_i == 3:
                run.font.color.rgb = C_TEAL
            elif c_i == 1:
                run.font.color.rgb = C_VERM
            else:
                run.font.color.rgb = C_TEXT

    # Right Card: IEEE References & Reproducibility Note (x = 286 .. 412, w = 126, h = 47)
    add_rect(slide, 286.0, 541.0, 126.0, 47.0, fill=C_WHITE, border=C_BORDER, border_pt=1.1)
    tb_ref = slide.shapes.add_textbox(Mm(288.5), Mm(542.5), Mm(121.0), Mm(44.0))
    tf_r = tb_ref.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Mm(0.5)

    p_rh = tf_r.paragraphs[0]
    p_rh.space_after = Pt(2.0)
    rrh = p_rh.add_run()
    rrh.text = "TÀI LIỆU THAM KHẢO (IEEE)"
    rrh.font.name = FONT_MAIN
    rrh.font.size = Pt(10.8)
    rrh.font.bold = True
    rrh.font.color.rgb = C_NAVY

    refs = [
        "[1] H. Zhuang et al., “F-OAL: Forward-only Online Analytic Learning...,” in NeurIPS, 2024.",
        "[2] Z. Huo et al., “Decoupled Parallel Backpropagation,” in ICML, 2018, pp. 2098–2106.",
        "[3] T. M. Cover, “Geometrical and Statistical Properties of Linear Inequalities,” IEEE Trans. EC, 1965.",
        "[4] V. A. Marchenko and L. A. Pastur, “Distribution of Eigenvalues...,” Math. USSR-Sb., 1967.",
        "[5] W&B Public Project: forge.coreweave.com/.../hcmus-foal-reproduction (tag: lab2-empirical-demo).",
    ]
    for r_txt in refs:
        pr = tf_r.add_paragraph()
        pr.space_after = Pt(1.4)
        pr.line_spacing = 1.06
        rr = pr.add_run()
        rr.text = r_txt
        rr.font.name = FONT_MAIN
        rr.font.size = Pt(8.9)
        rr.font.color.rgb = C_MUTED

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    return OUT_PPTX


def export_pptx_preview(pptx_path: Path, png_path: Path) -> bool:
    """Export an exact rasterized PNG preview via PowerPoint COM if installed on Windows."""
    ps_script = f"""
$ErrorActionPreference = 'Stop'
$ppt = New-Object -ComObject PowerPoint.Application
try {{
    $pres = $ppt.Presentations.Open('{str(pptx_path.resolve())}', $true, $false, $false)
    $pres.Slides.Item(1).Export('{str(png_path.resolve())}', 'PNG', 2480, 3508)
    $pres.Close()
}} finally {{
    $ppt.Quit()
}}
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=True,
            capture_output=True,
            timeout=45,
        )
        return png_path.exists()
    except Exception as exc:
        print(f"[Preview Info] PowerPoint COM export unavailable ({exc}); PPTX file itself is complete.")
        return False


def main() -> None:
    """Build Portrait A2 PPTX and verify slide count, dimensions, and file size."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = build_poster()
    size_mb = out.stat().st_size / (1024.0 * 1024.0)
    prs_check = Presentation(str(out))
    w_mm = prs_check.slide_width / 36000.0
    h_mm = prs_check.slide_height / 36000.0
    print(f"[Poster Built] {out.relative_to(REPO_ROOT).as_posix()}")
    print(f"  Slides     : {len(prs_check.slides)} (Required: 1)")
    print(f"  Dimensions : {w_mm:.1f} mm x {h_mm:.1f} mm (Portrait A2 = 420.0 mm x 594.0 mm)")
    print(f"  File size  : {size_mb:.2f} MB")
    if export_pptx_preview(out, OUT_PREVIEW):
        print(f"  Preview PNG: {OUT_PREVIEW.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
