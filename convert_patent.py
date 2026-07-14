import os
import re
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def md_to_docx(md_path, docx_path):
    doc = Document()
    
    # Set standard margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Styles config
    styles = doc.styles
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Calibri'
    normal_font.size = Pt(11)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_text = []
    
    # Simple table parser state
    in_table = False
    table_headers = []
    table_rows = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip('\n')
        
        # Code block handling
        if line.strip().startswith('```'):
            if in_code_block:
                # Close code block
                in_code_block = False
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.5)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                run = p.add_run('\n'.join(code_text))
                run.font.name = 'Consolas'
                run.font.size = Pt(9.5)
                # Add light gray background shading to code block
                shd = parse_xml(r'<w:shd {} w:fill="F4F4F4"/>'.format(nsdecls('w')))
                p._p.get_or_add_pPr().append(shd)
                code_text = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_text.append(line)
            i += 1
            continue
            
        # Table handling
        if line.strip().startswith('|'):
            in_table = True
            # Parse table line
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if i + 1 < len(lines) and lines[i+1].strip().startswith('|---'):
                # This is header
                table_headers = parts
                i += 2  # skip divider
                continue
            else:
                table_rows.append(parts)
                i += 1
                continue
        elif in_table:
            # End of table, write to docx
            if table_headers or table_rows:
                num_cols = len(table_headers) if table_headers else (len(table_rows[0]) if table_rows else 0)
                if num_cols > 0:
                    tbl = doc.add_table(rows=0, cols=num_cols)
                    tbl.style = 'Table Grid'
                    
                    if table_headers:
                        hdr_cells = tbl.add_row().cells
                        for col_idx, header_text in enumerate(table_headers):
                            hdr_cells[col_idx].text = header_text
                            # Bold header
                            for p in hdr_cells[col_idx].paragraphs:
                                for r in p.runs:
                                    r.font.bold = True
                                    
                    for row_data in table_rows:
                        row_cells = tbl.add_row().cells
                        # Fill in cells, truncate if data is longer or shorter than cols
                        for col_idx in range(min(num_cols, len(row_data))):
                            row_cells[col_idx].text = row_data[col_idx]
                            
                doc.add_paragraph()  # spacing
            in_table = False
            table_headers = []
            table_rows = []
            
        # Heading 1
        if line.startswith('# '):
            text = line[2:].strip()
            p = doc.add_heading(text, level=1)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = 'Calibri Light'
                r.font.size = Pt(18)
                r.font.bold = True
                
        # Heading 2
        elif line.startswith('## '):
            text = line[3:].strip()
            p = doc.add_heading(text, level=2)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = 'Calibri Light'
                r.font.size = Pt(14)
                r.font.bold = True
                
        # Heading 3
        elif line.startswith('### '):
            text = line[4:].strip()
            p = doc.add_heading(text, level=3)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = 'Calibri Light'
                r.font.size = Pt(12)
                r.font.bold = True
                
        # Bullet list item
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            text = line.strip()[2:].strip()
            # Parse inline bold formatting like **text**
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(3)
            _add_formatted_runs(p, text)
            
        # Numbered list item
        elif re.match(r'^\d+\.\s', line.strip()):
            match = re.match(r'^(\d+)\.\s(.*)', line.strip())
            text = match.group(2).strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(3)
            _add_formatted_runs(p, text)
            
        # Normal paragraph (ignore empty lines)
        elif line.strip():
            text = line.strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            _add_formatted_runs(p, text)
            
        i += 1
        
    doc.save(docx_path)
    print(f"Successfully converted {md_path} to {docx_path}")

def _add_formatted_runs(paragraph, text):
    # Matches markdown bold **text** or __text__
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.font.bold = True
        elif part.startswith('*') and part.endswith('*'):
            run = paragraph.add_run(part[1:-1])
            run.font.italic = True
        else:
            # Handle links like [text](link) - clean up link notation
            clean_part = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', part)
            paragraph.add_run(clean_part)

if __name__ == '__main__':
    md_file = 'manasitra/PATENT_TECHNICAL_DISCLOSURE.md'
    docx_file = 'manasitra/PATENT_TECHNICAL_DISCLOSURE.docx'
    md_to_docx(md_file, docx_file)
