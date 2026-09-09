# -*- coding: utf-8 -*-
# 실험결과_보고서.md -> 실험결과_보고서.pdf 변환 (일회성 유틸리티)
# reportlab만 사용 (pandoc에 PDF 엔진이 없어서 직접 구현)

import os
import re
import html
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image as RLImage, HRFlowable, ListFlowable, ListItem,
                                 Preformatted, PageBreak, KeepTogether)
from reportlab.lib.utils import ImageReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "실험결과_보고서.md")
PDF_PATH = os.path.join(BASE_DIR, "실험결과_보고서.pdf")

# ── 한글 지원 CID 폰트 등록 (외부 폰트 파일 불필요, reportlab 내장) ──
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))   # 명조 계열 (본문)
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))      # 고딕 계열 (제목/굵게)
FONT_BODY = "HYSMyeongJo-Medium"
FONT_BOLD = "HYGothic-Medium"

styles = {
    "h1": ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=20, leading=26,
                          spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#1a1a1a")),
    "h2": ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=15, leading=20,
                          spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a3a6b")),
    "h3": ParagraphStyle("h3", fontName=FONT_BOLD, fontSize=12.5, leading=17,
                          spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2a4a7b")),
    "h4": ParagraphStyle("h4", fontName=FONT_BOLD, fontSize=10.5, leading=14,
                          spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#3a5a8b")),
    "body": ParagraphStyle("body", fontName=FONT_BODY, fontSize=9.5, leading=15,
                            spaceBefore=2, spaceAfter=6, alignment=TA_LEFT),
    "quote": ParagraphStyle("quote", fontName=FONT_BODY, fontSize=9.5, leading=14,
                             spaceBefore=4, spaceAfter=8, leftIndent=14,
                             textColor=colors.HexColor("#444444"),
                             borderColor=colors.HexColor("#bbbbbb"), borderWidth=0,
                             backColor=colors.HexColor("#f4f4f4")),
    "caption": ParagraphStyle("caption", fontName=FONT_BODY, fontSize=8, leading=11,
                               spaceBefore=2, spaceAfter=10, textColor=colors.HexColor("#666666")),
    "table_cell": ParagraphStyle("table_cell", fontName=FONT_BODY, fontSize=8, leading=11),
    "table_head": ParagraphStyle("table_head", fontName=FONT_BOLD, fontSize=8, leading=11,
                                  textColor=colors.white),
    "bullet": ParagraphStyle("bullet", fontName=FONT_BODY, fontSize=9.5, leading=14,
                              spaceBefore=1, spaceAfter=1, leftIndent=10),
}

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 30 * mm


def inline_md(text):
    """인라인 마크다운(굵게/코드/이미지 alt 등)을 reportlab 미니 XML로 변환"""
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r'<font face="%s">\1</font>' % FONT_BOLD, text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="8.5" color="#a03030">\1</font>', text)
    return text


def flow_paragraph(text, style="body"):
    return Paragraph(inline_md(text), styles[style])


def parse_table(lines):
    """마크다운 표 블록(헤더+구분선+데이터행)을 reportlab Table로 변환"""
    rows = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 2:
        return None
    grid = []
    for i, row in enumerate(rows):
        if i == 1:
            continue  # 구분선(---) 행은 건너뜀
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        grid.append(cells)
    ncols = max(len(r) for r in grid)
    for r in grid:
        while len(r) < ncols:
            r.append("")

    is_image_table = any("![" in c for row in grid for c in row)

    data = []
    for ridx, row in enumerate(grid):
        out_row = []
        for c in row:
            m = re.match(r"!\[(.*?)\]\((.*?)\)", c.strip())
            if m:
                alt, path = m.group(1), m.group(2)
                full = os.path.join(BASE_DIR, path)
                if os.path.exists(full):
                    try:
                        ir = ImageReader(full)
                        iw, ih = ir.getSize()
                        max_w = (CONTENT_W / max(ncols, 1)) - 6
                        scale = max_w / iw
                        out_row.append(RLImage(full, width=iw * scale, height=ih * scale))
                    except Exception:
                        out_row.append(Paragraph(inline_md(alt), styles["table_cell"]))
                else:
                    out_row.append(Paragraph(inline_md(alt or "(그림 없음)"), styles["table_cell"]))
            else:
                st = "table_head" if ridx == 0 else "table_cell"
                out_row.append(Paragraph(inline_md(c), styles[st]))
        data.append(out_row)

    col_w = CONTENT_W / ncols
    t = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if not is_image_table:
        style_cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4a7b")))
        style_cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1),
                            [colors.white, colors.HexColor("#f0f4fa")]))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_story(md_text):
    lines = md_text.split("\n")
    story = []
    i = 0
    n = len(lines)
    in_code = False
    code_buf = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 코드펜스(``` ... ```) - ASCII 아트 그림 placeholder 등
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                if code_buf:
                    pre = Preformatted("\n".join(code_buf), ParagraphStyle(
                        "code", fontName="Courier", fontSize=6.8, leading=8.5,
                        textColor=colors.HexColor("#555555")))
                    story.append(pre)
                    story.append(Spacer(1, 6))
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped == "":
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                     color=colors.HexColor("#cccccc")))
            story.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline_md(stripped[2:]), styles["h1"]))
            i += 1
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(inline_md(stripped[3:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("#### "):
            story.append(Paragraph(inline_md(stripped[5:]), styles["h4"]))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline_md(stripped[4:]), styles["h3"]))
            i += 1
            continue

        # 표
        if stripped.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            tbl = parse_table(block)
            if tbl is not None:
                story.append(tbl)
                story.append(Spacer(1, 10))
            continue

        # 단독 이미지 줄: ![alt](path)
        m_img = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", stripped)
        if m_img:
            alt, path = m_img.group(1), m_img.group(2)
            full = os.path.join(BASE_DIR, path)
            if os.path.exists(full):
                ir = ImageReader(full)
                iw, ih = ir.getSize()
                scale = min(CONTENT_W / iw, 1.0)
                story.append(RLImage(full, width=iw * scale, height=ih * scale))
                if alt:
                    story.append(Paragraph(inline_md(alt), styles["caption"]))
            else:
                story.append(Paragraph("[그림 누락: %s]" % path, styles["caption"]))
            i += 1
            continue

        # 인용(blockquote), 여러 줄 연속 지원
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline_md(" ".join(buf)), styles["quote"]))
            continue

        # 불릿 리스트
        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            while i < n and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                txt = lines[i].strip()[2:]
                items.append(ListItem(Paragraph(inline_md(txt), styles["bullet"]),
                                       leftIndent=12, bulletColor=colors.HexColor("#2a4a7b")))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="•",
                                       leftIndent=14, spaceBefore=2, spaceAfter=8))
            continue

        # 일반 문단 (다음 빈 줄/특수줄까지 묶기)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() != "" and not lines[i].strip().startswith(("#", "|", ">", "```", "---", "- ", "* ")) \
                and not re.match(r"^!\[", lines[i].strip()):
            buf.append(lines[i])
            i += 1
        story.append(flow_paragraph(" ".join(s.strip() for s in buf), "body"))

    return story


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="APEN 실험결과 보고서",
    )
    story = build_story(md_text)
    doc.build(story)
    print("saved:", PDF_PATH)
    print("pages(approx via size):", os.path.getsize(PDF_PATH), "bytes")


if __name__ == "__main__":
    main()
