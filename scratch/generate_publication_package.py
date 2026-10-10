import os
import re
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

# Define paths
ROOT_DIR = r"d:\Rag book work"
PROCESSED_DIR = r"d:\Rag book work\data\processed\rag_baseline"
PUB_DIR = os.path.join(ROOT_DIR, "data", "processed", "publication")
os.makedirs(PUB_DIR, exist_ok=True)

MANUSCRIPT_MD_PATH = os.path.join(PROCESSED_DIR, "rag_book_final_manuscript.md")
DOCX_PATH = os.path.join(PUB_DIR, "rag_book_final_manuscript.docx")
PDF_PATH = os.path.join(PUB_DIR, "rag_book_final_manuscript.pdf")
README_PATH = os.path.join(ROOT_DIR, "README.md")

print(f"Reading manuscript from: {MANUSCRIPT_MD_PATH}")
with open(MANUSCRIPT_MD_PATH, "r", encoding="utf-8") as f:
    manuscript_text = f.read()

# ---------------------------------------------------------
# Markdown Parser Helper Functions
# ---------------------------------------------------------

def sanitize_xml_string(text):
    if not text:
        return ""
    # Remove control characters invalid in XML 1.0 (keep \n \r \t)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(text))

def clean_inline_markdown(text):
    """Removes or converts inline markdown formatting for plain/styled text."""
    text = sanitize_xml_string(text)
    # Convert file links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Convert math inline $...$ or \(...\) -> text
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    return text

def parse_markdown_blocks(md_text):
    lines = md_text.splitlines()
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1 # skip closing ```
            blocks.append({
                "type": "code",
                "lang": lang,
                "content": "\n".join(code_lines)
            })
            continue

        # Table block
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append({
                "type": "table",
                "lines": table_lines
            })
            continue

        # Callout / Alert block
        if stripped.startswith("> [!"):
            callout_type = stripped[4:stripped.find("]")].strip()
            alert_lines = [stripped[stripped.find("]")+1:].strip()]
            i += 1
            while i < n and lines[i].strip().startswith(">"):
                alert_lines.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append({
                "type": "callout",
                "alert_type": callout_type,
                "content": " ".join(filter(None, alert_lines))
            })
            continue

        # Blockquote
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append({
                "type": "blockquote",
                "content": " ".join(quote_lines)
            })
            continue

        # Headings
        if stripped.startswith("#"):
            level = 0
            while level < len(stripped) and stripped[level] == "#":
                level += 1
            title = stripped[level:].strip()
            blocks.append({
                "type": "heading",
                "level": level,
                "text": title
            })
            i += 1
            continue

        # Horizontal Rule
        if stripped in ["---", "***", "___"]:
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Lists
        if stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append({
                "type": "bullet",
                "text": stripped[2:].strip()
            })
            i += 1
            continue

        m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m_num:
            blocks.append({
                "type": "numbered",
                "number": m_num.group(1),
                "text": m_num.group(2).strip()
            })
            i += 1
            continue

        # Regular Paragraph
        blocks.append({
            "type": "paragraph",
            "text": stripped
        })
        i += 1

    return blocks

print("Parsing markdown blocks...")
parsed_blocks = parse_markdown_blocks(manuscript_text)
print(f"Extracted {len(parsed_blocks):,} blocks.")

# ---------------------------------------------------------
# DOCX Generator
# ---------------------------------------------------------

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def generate_docx(blocks, output_path):
    print(f"Generating DOCX at: {output_path}")
    doc = docx.Document()

    # Set page margins (1 inch all sides)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Palette
    NAVY = RGBColor(26, 54, 93)      # #1A365D
    SLATE = RGBColor(43, 108, 176)   # #2B6CB0
    CHARCOAL = RGBColor(45, 55, 72)  # #2D3748
    GREY_BG = "F7FAFC"

    # Style base
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = CHARCOAL

    for b in blocks:
        btype = b["type"]

        if btype == "heading":
            level = b["level"]
            text = clean_inline_markdown(b["text"])

            if level == 1:
                # Add page break before Major Parts & Chapters
                if text.startswith("Chapter ") or text.startswith("Part ") or text in ["Table of Contents", "List of Figures", "List of Tables", "References & Primary Source Links", "Technical Glossary", "Back Matter"]:
                    doc.add_page_break()
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(16)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.keep_with_next = True
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(20)
                run.font.bold = True
                run.font.color.rgb = NAVY
            elif level == 2:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.keep_with_next = True
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = SLATE
            elif level == 3:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(12)
                run.font.bold = True
                run.font.color.rgb = CHARCOAL
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = True
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(11)
                run.font.bold = True
                run.font.color.rgb = CHARCOAL

        elif btype == "paragraph":
            text = clean_inline_markdown(b["text"])
            if text.startswith("*Figure ") or text.startswith("Figure "):
                # Figure caption
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(10)
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(9.5)
                run.font.italic = True
                run.font.color.rgb = RGBColor(113, 128, 150)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = 1.15
                run = p.add_run(text)
                run.font.name = 'Calibri'
                run.font.size = Pt(10.5)

        elif btype == "bullet":
            text = clean_inline_markdown(b["text"])
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(text)
            run.font.name = 'Calibri'
            run.font.size = Pt(10)

        elif btype == "numbered":
            text = clean_inline_markdown(b["text"])
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(text)
            run.font.name = 'Calibri'
            run.font.size = Pt(10)

        elif btype == "code":
            table = doc.add_table(rows=1, cols=1)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell = table.cell(0, 0)
            cell.width = Inches(6.5)
            set_cell_background(cell, "F7FAFC")
            set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(sanitize_xml_string(b["content"]))
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(45, 55, 72)
            
            # Add spacer paragraph after table
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(6)

        elif btype == "callout":
            table = doc.add_table(rows=1, cols=1)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell = table.cell(0, 0)
            cell.width = Inches(6.5)
            
            # Left border color based on callout type
            bg_color = "EDF2F7"
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            run_title = p.add_run(f"[{b['alert_type']}] ")
            run_title.font.name = 'Calibri'
            run_title.font.bold = True
            run_title.font.size = Pt(10)
            run_title.font.color.rgb = NAVY
            
            run_text = p.add_run(clean_inline_markdown(b["content"]))
            run_text.font.name = 'Calibri'
            run_text.font.size = Pt(10)
            
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(6)

        elif btype == "blockquote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(clean_inline_markdown(b["content"]))
            run.font.name = 'Calibri'
            run.font.italic = True
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(74, 85, 104)

        elif btype == "table":
            lines = b["lines"]
            # Parse markdown pipe table
            rows_data = []
            for l in lines:
                parts = [p.strip() for p in l.strip("|").split("|")]
                # Filter out separator row like |---|---|
                if all(re.match(r'^:?-+:?$', p) for p in parts):
                    continue
                rows_data.append(parts)

            if rows_data:
                num_rows = len(rows_data)
                num_cols = max(len(r) for r in rows_data)
                table = doc.add_table(rows=num_rows, cols=num_cols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = True

                for r_idx, r_data in enumerate(rows_data):
                    row = table.rows[r_idx]
                    is_header = (r_idx == 0)
                    for c_idx, c_text in enumerate(r_data):
                        if c_idx < len(row.cells):
                            cell = row.cells[c_idx]
                            cell.text = clean_inline_markdown(c_text)
                            p = cell.paragraphs[0]
                            p.paragraph_format.space_after = Pt(0)
                            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
                            
                            if is_header:
                                set_cell_background(cell, "1A365D")
                                for run in p.runs:
                                    run.font.name = 'Calibri'
                                    run.font.bold = True
                                    run.font.size = Pt(9.5)
                                    run.font.color.rgb = RGBColor(255, 255, 255)
                            else:
                                if r_idx % 2 == 1:
                                    set_cell_background(cell, "F7FAFC")
                                else:
                                    set_cell_background(cell, "FFFFFF")
                                for run in p.runs:
                                    run.font.name = 'Calibri'
                                    run.font.size = Pt(9)

                sp = doc.add_paragraph()
                sp.paragraph_format.space_after = Pt(6)

        elif btype == "hr":
            sp = doc.add_paragraph()
            sp.paragraph_format.space_before = Pt(6)
            sp.paragraph_format.space_after = Pt(6)

    doc.save(output_path)
    print(f"Successfully generated DOCX at: {output_path}")

# ---------------------------------------------------------
# ReportLab PDF Generator
# ---------------------------------------------------------

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            # Suppress header/footer on title cover page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header
        self.drawString(54, 750, "Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)

        # Footer
        self.line(54, 50, 558, 50)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.drawString(54, 38, "Antigravity AI Engineering & Research Team | Confidential & Published")
        self.restoreState()


def generate_pdf(blocks, output_path):
    print(f"Generating PDF at: {output_path}")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#1A365D")
    SLATE = colors.HexColor("#2B6CB0")
    CHARCOAL = colors.HexColor("#2D3748")

    # Modify/Add Paragraph Styles
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].fontSize = 9.5
    styles['Normal'].leading = 13
    styles['Normal'].textColor = CHARCOAL

    style_h1 = ParagraphStyle(
        'CustomH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'CustomH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=SLATE,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h3 = ParagraphStyle(
        'CustomH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=CHARCOAL,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        spaceAfter=6
    )

    style_bullet = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        leftIndent=15,
        spaceAfter=3
    )

    style_caption = ParagraphStyle(
        'CustomCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1, # Center
        textColor=colors.HexColor("#718096"),
        spaceBefore=4,
        spaceAfter=8
    )

    style_code = ParagraphStyle(
        'CustomCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=CHARCOAL
    )

    style_callout = ParagraphStyle(
        'CustomCallout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=CHARCOAL
    )

    story = []

    for b in blocks:
        btype = b["type"]

        if btype == "heading":
            level = b["level"]
            text = clean_inline_markdown(b["text"])

            if level == 1:
                if text.startswith("Chapter ") or text.startswith("Part ") or text in ["Table of Contents", "List of Figures", "List of Tables", "References & Primary Source Links", "Technical Glossary", "Back Matter"]:
                    story.append(PageBreak())
                story.append(Paragraph(text, style_h1))
            elif level == 2:
                story.append(Paragraph(text, style_h2))
            elif level == 3:
                story.append(Paragraph(text, style_h3))
            else:
                story.append(Paragraph(text, style_h3))

        elif btype == "paragraph":
            text = clean_inline_markdown(b["text"])
            if text.startswith("*Figure ") or text.startswith("Figure "):
                story.append(Paragraph(text, style_caption))
            else:
                story.append(Paragraph(text, style_body))

        elif btype == "bullet" or btype == "numbered":
            text = clean_inline_markdown(b["text"])
            story.append(Paragraph(f"• {text}", style_bullet))

        elif btype == "code":
            code_text = sanitize_xml_string(b["content"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            p_code = Preformatted(code_text, style_code)
            t_code = Table([[p_code]], colWidths=[504])
            t_code.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_code)
            story.append(Spacer(1, 6))

        elif btype == "callout":
            callout_text = f"<b>[{b['alert_type']}]</b> {clean_inline_markdown(b['content'])}"
            p_call = Paragraph(callout_text, style_callout)
            t_call = Table([[p_call]], colWidths=[504])
            t_call.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
                ('LINELEFT', (0,0), (0,0), 3, NAVY),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                ('PADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(t_call)
            story.append(Spacer(1, 6))

        elif btype == "blockquote":
            p_quote = Paragraph(f"<i>{clean_inline_markdown(b['content'])}</i>", style_body)
            t_quote = Table([[p_quote]], colWidths=[480])
            t_quote.setStyle(TableStyle([
                ('LINELEFT', (0,0), (0,0), 2, colors.HexColor("#A0AEC0")),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t_quote)
            story.append(Spacer(1, 6))

        elif btype == "table":
            lines = b["lines"]
            rows_data = []
            for l in lines:
                parts = [p.strip() for p in l.strip("|").split("|")]
                if all(re.match(r'^:?-+:?$', p) for p in parts):
                    continue
                rows_data.append(parts)

            if rows_data:
                num_cols = max(len(r) for r in rows_data)
                col_width = 504 / num_cols
                
                table_cells = []
                for r_idx, r_data in enumerate(rows_data):
                    row_cells = []
                    is_header = (r_idx == 0)
                    for c_text in r_data:
                        c_clean = clean_inline_markdown(c_text)
                        if is_header:
                            p_cell = Paragraph(f"<b><font color='white'>{c_clean}</font></b>", styles['Normal'])
                        else:
                            p_cell = Paragraph(c_clean, styles['Normal'])
                        row_cells.append(p_cell)
                    # Pad missing columns if any
                    while len(row_cells) < num_cols:
                        row_cells.append(Paragraph("", styles['Normal']))
                    table_cells.append(row_cells)

                t_table = Table(table_cells, colWidths=[col_width]*num_cols)
                t_style = [
                    ('BACKGROUND', (0,0), (-1,0), NAVY),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                    ('PADDING', (0,0), (-1,-1), 4),
                ]
                for r in range(1, len(table_cells)):
                    if r % 2 == 1:
                        t_style.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor("#F7FAFC")))
                t_table.setStyle(TableStyle(t_style))
                story.append(t_table)
                story.append(Spacer(1, 6))

        elif btype == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceBefore=6, spaceAfter=6))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {output_path}")

# ---------------------------------------------------------
# Update README.md
# ---------------------------------------------------------

README_CONTENT = """# RAG Book: Retrieval-Augmented Generation — From Baseline Pipelines to Production Systems

Welcome to the official repository for **"Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems (An Empirical Engineering Guide to Component Evaluation, Infrastructure Architecture, and Adaptive Retrieval)"**.

---

## 📌 Executive Summary & Key Contributions

This project provides an empirically audited, end-to-end research and engineering guide to Retrieval-Augmented Generation (RAG). Built around the multi-hop question answering benchmark **HotpotQA** (`BeIR/hotpotqa`), this codebase evaluates 31 distinct pipeline configurations across 13 experimental phases—culminating in a leakage-safe held-out benchmark split with empirical 95% bootstrap confidence intervals.

### Primary Engineering Takeaways:
1. **Generator Capacity Bounds Performance:** Perfect retrieval recall (Recall@5 = 1.0000) is insufficient if the generator model lacks parametric reasoning capacity. Generator model scaling yielded the single largest accuracy gain in our campaign (+12.0% gain over baseline).
2. **Cross-Encoder Reranking Delivers High Precision:** Reranking over a hybrid candidate pool ($K_{\\text{init}}=15$) drove top-5 retrieval recall to **100.0%** while adding less than 800 ms of online latency.
3. **Query Transformations Can Degrade Accuracy:** Small LLM query rewriters, multi-query expansion pipelines, and hypothetical document generation (HyDE) introduce semantic drift and hallucinations, reducing end-to-end answer accuracy by 4.0% to 10.0%.
4. **Contextual Document Summaries Are Cost-Inefficient:** Prepending LLM-generated document summaries to text chunks added **1,220 seconds** of offline precomputation time without improving answer accuracy (0.0% gain, -2.0% Exact Match regression due to token dilution).
5. **Generation Dominates System Latency:** In a production RAG pipeline, LLM text generation accounts for **88.5% of total online response latency** (6,436.84 ms out of 7,272.35 ms mean), dwarfing dense retrieval (0.6%) and cross-encoder reranking (10.8%).

---

## 📊 Authoritative Final Benchmark Results (`PROTO_FINAL_BENCHMARK`)

Evaluated on the held-out **40-query Test Set** ($N=40$) under frozen hyperparameters (seed `42`, greedy decoding $T=0.0$):

| Performance Metric | Point Estimate ($N=40$ Test) | Empirical 95% Bootstrap Confidence Interval | Dev Set Result ($N=10$) |
| :--- | :---: | :---: | :---: |
| **Answer Accuracy (%)** | **90.00%** (36/40) | **[80.00%, 97.50%]** | 100.00% (10/10) |
| **Exact Match Rate (%)** | **85.00%** (34/40) | **[72.50%, 95.00%]** | 80.00% (8/10) |
| **Hit Rate @ 5 (%)** | **100.00%** (40/40) | **[100.00%, 100.00%]** | 100.00% (10/10) |
| **Recall @ 5** | **1.0000** | **[1.0000, 1.0000]** | 1.0000 |
| **Precision @ 5** | **0.4000** | **[0.4000, 0.4000]** | 0.4000 |
| **Total Online Response Latency**| **7,272.35 ms** (7.27 s) | **[6,768.42 ms, 7,874.62 ms]** | 9,541.95 ms |
| **LLM Generation Latency Share**| **6,436.84 ms** (88.5%) | **[5,959.18 ms, 7,021.75 ms]** | - |
| **Total Offline Indexing Time** | **13.3266 s** | - | - |

*Failure Breakdown:* 0 Retrieval Failures (`RF`), 3 Multi-Hop Reasoning Failures (`MH`), 0 Generation Failures (`GF`), 0 Insufficient Context (`IC`), 1 Data-Quality Issue (`DQ` on QID `5adf48b25542995ec70e8fb5`).

---

## 📁 Repository & Code Structure

```text
d:\Rag book work\
├── README.md                   # Main project overview & benchmark documentation
├── requirements.txt            # Python dependencies
├── config/
│   └── config.yaml             # Dataset HF paths, random seed 42, and baseline parameters
├── data/
│   ├── processed/
│   │   ├── hotpotqa_pilot_queries.json
│   │   ├── hotpotqa_pilot_corpus.json
│   │   ├── hotpotqa_pilot_qrels.json
│   │   ├── chapter_7_ablation/         # Master results JSON & ablation logs
│   │   ├── final_benchmark/           # Authoritative benchmark JSON, logs, & CIs
│   │   ├── publication/               # Publication package (DOCX, PDF)
│   │   ├── full_book_final_audit.md   # Complete manuscript audit report
│   │   ├── rag_book_publication_qa.md # Publication QA checklist
│   │   └── rag_book_final_manuscript.md # Assembled manuscript source
│   └── manifest.json           # Reproducible dataset schema manifest
├── src/
│   ├── data/
│   │   ├── loader.py           # PyArrow streaming loader & schema inspection
│   │   └── manifest.py         # Manifest validator
│   ├── rag/
│   │   ├── chunker.py          # Sliding window text chunking (250w/50o)
│   │   ├── retriever.py        # FAISS dense retriever & evaluation metrics
│   │   └── generator.py        # FLAN-T5 generator wrapper (greedy decoding)
│   └── utils/
│       └── logging_config.py   # Microsecond timing and logger setup
└── scripts/                    # Reproducible experiment execution scripts (Phases 1--13)
```

---

## 🚀 How to Reproduce Benchmark Results

### 1. Environment Setup
```powershell
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install python-docx reportlab markdown
```

### 2. Stream Dataset & Execute Pipelines
```powershell
# Stream BeIR/hotpotqa dataset via PyArrow & build manifest
.\.venv\Scripts\python.exe scripts/load_pilot_dataset.py

# Execute Chapter 7 Ablation Campaign (Runs 0--4)
.\.venv\Scripts\python.exe scripts/run_hybrid_experiment.py
```

### 3. Generate Manuscript & Publication Package
```powershell
# Assemble final manuscript and export DOCX & PDF packages
.\.venv\Scripts\python.exe scratch/build_full_manuscript.py
.\.venv\Scripts\python.exe scratch/generate_publication_package.py
```

---

## 📖 Book Structure (Chapters 1–12)

- **Part I: Foundations & Dataset Schema**
  - Chapter 1: The RAG Dilemma & Empirical Foundations
  - Chapter 2: Dataset Architecture, Schemas, & Linked Sampling
  - Chapter 3: Building the Baseline Fixed-Pipeline RAG System
- **Part II: Core Retrieval Components & Exploratory Pilots**
  - Chapter 4: Granularity Matters: Document Chunking and Retrieval Depth
  - Chapter 5: Embedding Models, Lexical Search, & Reranking
  - Chapter 6: Advanced Retrieval Extensions: Query Transformations & Context Augmentation
- **Part III: Controlled Ablations & Authoritative Benchmarks**
  - Chapter 7: What Actually Matters? System-Level Lessons from Controlled Ablations and Benchmark Audits
- **Part IV: Architecture, Infrastructure, & Future Paradigms**
  - Chapter 8: Vector Databases and Frameworks: Architecture, Trade-Offs, Abstraction Leakage, and Operational Realities
  - Chapter 9: Graph RAG: When Relationships Matter More Than Similarity
  - Chapter 10: Agentic RAG: Adaptive Retrieval, Planning, and Tool Use
  - Chapter 11: Production RAG: Reliability, Evaluation, Cost, and Deployment
  - Chapter 12: Future of RAG: Adaptive Architectures, Unresolved Frontiers, and the Engineering Discipline

---

## 📄 Citation Information

If you utilize this repository, evaluation framework, or empirical findings in your research or production systems, please cite as:

```bibtex
@book{antigravity2026rag,
  title     = {Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems},
  subtitle  = {An Empirical Engineering Guide to Component Evaluation, Infrastructure Architecture, and Adaptive Retrieval},
  author    = {Antigravity AI Engineering \& Research Team},
  year      = {2026},
  publisher = {Antigravity AI Press},
  url       = {d:/Rag book work/data/processed/publication/}
}
```
"""

def update_readme():
    print(f"Updating README.md at: {README_PATH}")
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)
    print("README.md updated successfully.")

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
if __name__ == "__main__":
    generate_docx(parsed_blocks, DOCX_PATH)
    generate_pdf(parsed_blocks, PDF_PATH)
    update_readme()
    print("Publication package generation complete!")
