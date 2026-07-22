import os
import sys
from pathlib import Path
from fpdf import FPDF

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class PDF(FPDF):
    def header(self):
        # Header on every page (except first/cover page if desired)
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.cell(w=self.epw/2, h=10, txt='Clash Royale Matchup Predictor Technical Report', align='L')
            self.cell(w=self.epw/2, h=10, txt=f'Page {self.page_no()}', align='R', new_x="LMARGIN", new_y="NEXT")
            self.line(10, 18, 200, 18)
            self.ln(5)

    def footer(self):
        # Footer on every page
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(w=self.epw, h=10, txt='[AUTHOR NAME]', align='C')

def clean_text(text: str) -> str:
    """Replaces non-latin or special math characters with plain text fallbacks to prevent FPDF unicode errors."""
    replacements = {
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2192": "->",
        "\u2215": "/",
        "\u03c3": "sigma",
        "\u03b8": "theta",
        "\u0394": "Delta",
        "\u00b1": "+/-",
        "\u2248": "~",
        "\u221e": "inf",
        "\u2260": "!=",
        "\u221d": "prop",
        "\u2229": "intersection",
        "\u222a": "union",
        "\u201d": '"',
        "\u201c": '"',
        "\u2019": "'",
        "\u2018": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\\ge": ">=",
        "\\le": "<=",
        "\\sigma": "sigma",
        "\\theta": "theta",
        "\\Delta": "Delta",
        "\\mathcal{L}": "L",
        "\\approx": "~",
        "\\cup": "union",
        "\\cap": "intersection",
        "\\in": "in",
        "\\times": "x",
        "\\cdot": "*",
        "\\to": "->",
        "\\log": "log",
        "\\sum": "sum",
        "\\frac": "",
        "\\left": "",
        "\\right": "",
        "_{deck}": "_deck",
        "_{proj}": "_proj",
        "\\_": "_",
        "$$": "",
        "$": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf():
    md_path = PROJECT_ROOT / "docs" / "paper.md"
    pdf_path = PROJECT_ROOT / "docs" / "paper.pdf"
    
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        sys.exit(1)
        
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.ln(40)
    title = clean_text("Modeling Intrinsic Matchup Advantages in Clash Royale")
    pdf.multi_cell(w=pdf.epw, h=10, txt=title, align='C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.ln(10)
    subtitle = clean_text("via Permutation-Invariant Deep Sets & Skew-Symmetric Heads")
    pdf.multi_cell(w=pdf.epw, h=10, txt=subtitle, align='C')
    
    pdf.ln(30)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(w=pdf.epw, h=10, txt='Prepared by [AUTHOR NAME]', align='C')
    pdf.multi_cell(w=pdf.epw, h=10, txt='Date: July 18, 2026', align='C')
    pdf.multi_cell(w=pdf.epw, h=10, txt='Technical Research Report', align='C')
    
    # Body Page
    pdf.add_page()
    pdf.set_font('Helvetica', '', 10)
    
    in_abstract = False
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            pdf.ln(3)
            continue
            
        # Parse abstract section
        if line_str.startswith("### Abstract"):
            in_abstract = True
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(w=pdf.epw, h=10, txt='Abstract', new_x="LMARGIN", new_y="NEXT", align='L')
            pdf.set_font('Helvetica', 'I', 10)
            continue
            
        # Check if abstract ended
        if in_abstract and line_str.startswith("---"):
            in_abstract = False
            pdf.set_font('Helvetica', '', 10)
            pdf.ln(5)
            continue
            
        # Skip table separators
        if line_str.startswith("|") and "-" in line_str and ":" in line_str:
            continue
            
        # Clean up table rows
        if line_str.startswith("|"):
            line_str = line_str.replace("|", "  ").strip()
            
        # Headers
        if line_str.startswith("# "):
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(w=pdf.epw, h=10, txt=clean_text(line_str[2:]), new_x="LMARGIN", new_y="NEXT", align='L')
            pdf.set_font('Helvetica', '', 10)
            continue
        elif line_str.startswith("## "):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(w=pdf.epw, h=10, txt=clean_text(line_str[3:]), new_x="LMARGIN", new_y="NEXT", align='L')
            pdf.set_font('Helvetica', '', 10)
            continue
        elif line_str.startswith("### "):
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(w=pdf.epw, h=10, txt=clean_text(line_str[4:]), new_x="LMARGIN", new_y="NEXT", align='L')
            pdf.set_font('Helvetica', '', 10)
            continue
            
        # Lists
        if line_str.startswith("* ") or line_str.startswith("- "):
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(w=pdf.epw, h=6, txt=f"  o  {clean_text(line_str[2:])}")
            continue
            
        # Standard paragraph
        pdf.set_font('Helvetica', '', 10)
        # Skip YAML frontmatter markers
        if line_str == "---":
            continue
        pdf.multi_cell(w=pdf.epw, h=6, txt=clean_text(line_str))
        
    pdf.output(str(pdf_path))
    print(f"\n[+] PDF compiled successfully at: {pdf_path}\n")

if __name__ == "__main__":
    generate_pdf()
