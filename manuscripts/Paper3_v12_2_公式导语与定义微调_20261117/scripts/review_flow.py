from pathlib import Path
from copy import deepcopy
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def bind_lead(doc, lab):
    """Use normal paragraph widow control to keep a final lead-in line with its equation."""
    mark=next(b for b in doc.element.body.iter(qn('w:bookmarkStart')) if b.get(qn('w:name'))=='PaperEq'+lab)
    eq=mark.getparent(); lead=eq.getprevious()
    if lead is None or lead.tag!=qn('w:p'):raise ValueError(lab)
    if not ''.join(t.text or '' for t in lead.iter(qn('w:t'))).rstrip().endswith('：'):raise ValueError(lab)
    before=[c for c in lead if c.tag!=qn('w:pPr')]
    br=OxmlElement('w:r');br.append(OxmlElement('w:br'))
    first=eq.find(qn('w:pPr')).getnext()
    for c in before:first.addprevious(c)
    first.addprevious(br)
    lead.getparent().remove(lead)
    p=Paragraph(eq,doc);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent=Pt(24)
    p.paragraph_format.space_before=Pt(0)
    p.paragraph_format.line_spacing=1.12


def bind_objective(doc):
    a=next(b.getparent() for b in doc.element.body.iter(qn('w:bookmarkStart')) if b.get(qn('w:name'))=='PaperEq28a')
    b=a.getnext()
    assert b.find(qn('w:bookmarkStart')).get(qn('w:name'))=='PaperEq28b'
    br=OxmlElement('w:r');br.append(OxmlElement('w:br'));a.append(br)
    for child in list(b):
        if child.tag!=qn('w:pPr'):a.append(child)
    b.getparent().remove(b)


def header_in_table(doc):
    for table in doc.tables[:3]:
        tab=table._tbl; cap=tab.getprevious()
        if cap is None or cap.tag!=qn('w:p'):continue
        text=''.join(t.text or '' for t in cap.iter(qn('w:t')))
        if not text.startswith('表'):continue
        row=table.add_row()
        tab.tr_lst[0].addprevious(row._tr)
        cell=row.cells[0].merge(row.cells[-1])
        for p in list(cell._tc.findall(qn('w:p'))):cell._tc.remove(p)
        cell._tc.append(cap)
        rp=row._tr.get_or_add_trPr()
        rp.append(OxmlElement('w:cantSplit'));rp.append(OxmlElement('w:tblHeader'))
        borders=OxmlElement('w:tcBorders')
        for side,val in [('top','nil'),('bottom','single')]:
            edge=OxmlElement('w:'+side);edge.set(qn('w:val'),val);edge.set(qn('w:sz'),'8');borders.append(edge)
        cell._tc.get_or_add_tcPr().append(borders)
        pp=Paragraph(cap,doc);pp.paragraph_format.space_before=Pt(2);pp.paragraph_format.space_after=Pt(2)



def split_at_sentence(doc, label, phrase):
    mark=next(b for b in doc.element.body.iter(qn('w:bookmarkStart')) if b.get(qn('w:name'))=='PaperEq'+label)
    last=mark.getparent().getprevious()
    new=OxmlElement('w:p')
    for r in last:
        if r.tag!=qn('w:r'):continue
        text=r.find(qn('w:t'))
        if text is None or phrase not in (text.text or ''):continue
        left,right=text.text.split(phrase,1)
        head=deepcopy(r);head.find(qn('w:t')).text=left+'。'
        text.text=phrase[1:]+right
        last.addprevious(new)
        for preceding in list(last)[:list(last).index(r)]:
            if preceding.tag==qn('w:pPr'):new.append(deepcopy(preceding))
            else:new.append(preceding)
        new.append(head)
        return
    raise ValueError((label,phrase))


def finish_pagination(doc):
    # A lead and its formula can use normal widow/orphan control in a single
    # paragraph. The soft line break keeps the mathematical display on its own line.
    split_at_sentence(doc,'25','。由RF')
    for label in ('5','11','36','25','31','B1','B5'):
        bind_lead(doc,label)
    bind_objective(doc)
    compat=doc.settings.element.find(qn('w:compat'))
    if compat.find(qn('w:doNotExpandShiftReturn')) is None:
        compat.insert(0,OxmlElement('w:doNotExpandShiftReturn'))
    header_in_table(doc)
    # At the final page boundaries these section headings start at the top.
    for p in doc.paragraphs:
        if p.text.startswith('IV. 预测驱动'):
            p.paragraph_format.space_before=Pt(26)
        if p.text in ('A. 导频观测与状态估计','B. 平均EE约束下的公平优化'):
            p.paragraph_format.space_before=Pt(55)
