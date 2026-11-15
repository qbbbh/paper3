from pathlib import Path
import os, re, subprocess, zipfile, json, unicodedata
from docx import Document
from docx.shared import Mm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.opc.constants import RELATIONSHIP_TYPE as RT


import sys
import tempfile
stage=Path(tempfile.gettempdir())/'Paper3_v9_Word_build'
stage.mkdir(exist_ok=True)
source=Path(__file__).resolve().parent.parent
annotated='--annotated' in sys.argv
inputfile=source/('02_中文全稿_v9_修改标注稿.md' if annotated else '01_中文全稿_v9_清洁稿.md')
output=source/('Paper3_中文全稿_v9_修改标注版.docx' if annotated else 'Paper3_中文全稿_v9_清洁审阅版.docx')
md=inputfile.read_text(encoding='utf-8-sig')
md=re.sub(r'<!-- CHANGE (M\d+) -->',r'CHANGEFLAG\1\n',md)
md=re.sub(r'!\[图(\d+)\]\(figures/F(\d+)\.svg\)\{width=100%\}',r'![](figures/F\2.png)',md)
md=re.sub(r'<sup>(\[\d+\])</sup>',lambda m: 'SUPCITE'+m[1][1:-1]+'ENDSUP',md)
md=re.sub(r'\{\\rm\s+([^{}]+)\}',r'{\\mathrm{\1}}',md)
md=re.sub(r'\\mathrm (\w)',r'\\mathrm{\1}',md)
def tag(m):
    eq=m[1]
    tags=re.findall(r'\\tag\{([^}]+)\}',eq)
    assert len(tags)<=1
    eq=re.sub(r'\\tag\{[^}]+\}', '', eq)
    return '$$'+eq+'$$'+ ('\n\nEQNUMBERFLAG'+tags[0] if tags else '')
md=re.sub(r'\$\$([\s\S]*?)\$\$',tag,md)

def algorithm(m):
    lines=[line for line in m[1].splitlines() if line and not line.startswith('──')]
    def notation(line):
        for old,new in [('R_min',r'$R_{\min}$'),('η_ref',r'$\eta_{\mathrm{ref}}$'),('p_keep',r'$\mathbf p^{\mathrm{keep}}$'),('pᵇ',r'$\mathbf p^{\mathrm b}$'),('x⁰',r'$\boldsymbol x^{(0)}$'),('p⁰',r'$\mathbf p^{(0)}$')]:
            line=line.replace(old,new)
        return line
    return '\n\n'.join('ALGORITHMLINEFLAG'+str(len(l)-len(l.lstrip()))+'TEXT'+notation(l.lstrip()) for l in lines)
md=re.sub(r'```text\n([\s\S]*?)\n```',algorithm,md)
(stage/'combined.md').write_text(md,encoding='utf-8')
raw=stage/'pandoc.docx'
cmd=[r'C:\Program Files\Pandoc\pandoc.exe','--from=markdown+tex_math_dollars+raw_html-smart-implicit_figures','--resource-path='+str(source),str(stage/'combined.md'),'-o',str(raw)]
proc=subprocess.run(cmd,cwd=source,capture_output=True,encoding='utf-8',check=True)
if proc.stderr.strip(): raise RuntimeError(proc.stderr)
d=Document(raw)
sec=d.sections[0]
sec.page_width=Mm(210); sec.page_height=Mm(297)
sec.left_margin=sec.right_margin=Mm(23)
sec.top_margin=Mm(24); sec.bottom_margin=Mm(23)
sec.header_distance=Mm(12); sec.footer_distance=Mm(12)
page_width=round((sec.page_width-sec.left_margin-sec.right_margin)/635)

for style in d.styles:
    if style.type==WD_STYLE_TYPE.TABLE:
        for border in style.element.xpath('.//w:tcBorders|.//w:tblBorders'):
            border.getparent().remove(border)
    if style.type in (WD_STYLE_TYPE.PARAGRAPH,WD_STYLE_TYPE.CHARACTER):
        style.font.size=Pt(12); style.font.name='Times New Roman'; style.font.color.rgb=RGBColor(0,0,0)
        rpr=style.element.get_or_add_rPr()
        fonts=rpr.rFonts
        for key in list(fonts.attrib):
            if key.rsplit('}',1)[-1].lower().endswith('theme'): del fonts.attrib[key]
        fonts.set(qn('w:eastAsia'),'宋体'); fonts.set(qn('w:cs'),'Times New Roman')
        for border in style.element.xpath('./w:pPr/w:pBdr'):
            border.getparent().remove(border)
normal=d.styles['Normal']
normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
normal.paragraph_format.line_spacing=1.18
normal.paragraph_format.space_after=Pt(4)
normal.paragraph_format.widow_control=True
for n in range(1,7):
    s=d.styles[f'Heading {n}']; s.font.bold=True
    fmt=s.paragraph_format; fmt.space_before=Pt(11 if n==1 else 8); fmt.space_after=Pt(4)
    fmt.keep_with_next=True; fmt.keep_together=True; fmt.line_spacing=1.12
for rpr in d.styles.element.xpath('./w:docDefaults/w:rPrDefault/w:rPr'):
    for fld,attrs in [('rFonts',{'ascii':'Times New Roman','hAnsi':'Times New Roman','eastAsia':'宋体','cs':'Times New Roman'}),('sz',{'val':'24'}),('szCs',{'val':'24'})]:
        node=rpr.find(qn(f'w:{fld}'))
        if node is None:
            node=OxmlElement(f'w:{fld}'); rpr.append(node)
        node.attrib.clear()
        for k,val in attrs.items(): node.set(qn(f'w:{k}'),val)

n_equations=0
eq_tables=[]
for index,p in enumerate(d.paragraphs):
    value=p.text.strip(); fmt=p.paragraph_format
    if index==0:
        p.style='Title'; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        fmt.space_after=Pt(14); fmt.keep_with_next=True
        for run in p.runs: run.bold=True
    elif value=='I. 引言':
        p.style='Heading 1'; fmt.space_before=Pt(0)
    elif value.startswith(('补充材料I　','补充材料II：','补充材料III：','补充材料IV：')):
        p.style='Heading 1'; fmt.page_break_before=value.startswith('补充材料I　')
        if fmt.page_break_before: fmt.space_before=Pt(0)
    elif p.style.name.startswith('Heading'):
        fmt.keep_with_next=True
    elif (mark:=re.fullmatch(r'EQNUMBERFLAG([A-Z]?\d+)',value)):
        q=p._p.getprevious()
        omml=q.xpath('./m:oMathPara/m:oMath')
        assert len(omml)==1,(value,q.xml[:400])
        eq_table=d.add_table(rows=1,cols=2)
        eq_table.autofit=False
        eq_tables.append(eq_table._tbl)
        eq_table.alignment=WD_TABLE_ALIGNMENT.CENTER
        eq_table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        pr=eq_table._tbl.tblPr
        dw=pr.find(qn('w:tblW')); dw.set(qn('w:type'),'dxa'); dw.set(qn('w:w'),str(page_width))
        for elem in pr.xpath('./w:tblBorders|./w:tblLook|./w:tblStyle'): elem.getparent().remove(elem)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','bottom','left','right','insideH','insideV']:
            border=OxmlElement(f'w:{edge}'); border.set(qn('w:val'),'nil'); borders.append(border)
        pr.append(borders)
        padding=OxmlElement('w:tblCellMar')
        for edge in ['top','bottom','left','right']:
            e=OxmlElement(f'w:{edge}'); e.set(qn('w:w'),'0'); e.set(qn('w:type'),'dxa'); padding.append(e)
        pr.append(padding)
        for j,w in enumerate([page_width-690,690]):
            eq_table.columns[j].width=Pt(w/20)
            c=eq_table.cell(0,j); c.width=Pt(w/20); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cb=OxmlElement('w:tcBorders')
            for edge in ['top','bottom','left','right','insideH','insideV']:
                border=OxmlElement(f'w:{edge}'); border.set(qn('w:val'),'nil'); cb.append(border)
            c._tc.get_or_add_tcPr().append(cb)
            ep=c.paragraphs[0]; ep.alignment=WD_ALIGN_PARAGRAPH.CENTER if j==0 else WD_ALIGN_PARAGRAPH.RIGHT
            ep.paragraph_format.space_before=Pt(2); ep.paragraph_format.space_after=Pt(5)
            ep.paragraph_format.keep_together=True; ep.paragraph_format.line_spacing=1.10
            if j==0:
                math=q.xpath('./m:oMathPara')[0]
                jc=math.find('.//'+qn('m:jc'))
                if jc is not None: jc.set(qn('m:val'),'center')
                ep._p.append(math)
            else: ep.add_run('('+mark[1]+')')
        q.addprevious(eq_table._tbl)
        q.getparent().remove(q); p._p.getparent().remove(p._p); n_equations+=1
    elif (line:=re.match(r'ALGORITHMLINEFLAG(\d+)TEXT',value)):
        for run in p.runs:
            if 'ALGORITHMLINEFLAG' in run.text:
                run.text=re.sub(r'ALGORITHMLINEFLAG\d+TEXT','',run.text)
        p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        fmt.left_indent=Pt(8+int(line[1])*4); fmt.line_spacing=1.05; fmt.space_before=Pt(0); fmt.space_after=Pt(0.7)
        fmt.first_line_indent=Pt(0)
        if re.match(r'\s*(3|13|14)\s',p.text): fmt.left_indent=Pt(22)
        fmt.keep_together=True
        if p.text.startswith(('算法1','第一阶段','第二阶段')):
            fmt.space_before=Pt(6); fmt.keep_with_next=True
            for run in p.runs: run.bold=True
    elif p._p.xpath('.//w:drawing'):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER; fmt.keep_with_next=True; fmt.space_after=Pt(3)
    elif re.match(r'^图[1-6]\s',value):
        p.alignment=WD_ALIGN_PARAGRAPH.LEFT; fmt.line_spacing=1.1; fmt.space_after=Pt(7); fmt.keep_together=True
    elif re.match(r'^表[IVX]+(?:-\d+)?\s+',value):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER; fmt.space_before=Pt(6); fmt.space_after=Pt(4); fmt.keep_with_next=True
    elif value.startswith('表注：'):
        p.alignment=WD_ALIGN_PARAGRAPH.LEFT; fmt.line_spacing=1.12; fmt.space_after=Pt(6)
    elif re.match(r'^\[\d+\]',value):
        fmt.left_indent=Pt(24); fmt.first_line_indent=Pt(-24); fmt.line_spacing=1.10
        fmt.space_after=Pt(4); fmt.keep_together=True
    elif p._p.xpath('./m:oMathPara'):
        fmt.keep_together=True; fmt.space_before=Pt(2); fmt.space_after=Pt(5)
    elif value:
        fmt.first_line_indent=Pt(24)
assert n_equations>=37,n_equations


def weight(s):
    return sum(1 if unicodedata.east_asian_width(c) in 'WFA' and ord(c)>127 else .53 for c in s)

for tab in d.tables:
    if tab._tbl in eq_tables: continue
    rows=tab.rows; ncol=len(tab.columns)
    headers=[c.text for c in rows[0].cells]
    if headers==['物理意义','符号','数值'] and len(rows)==4: ratio=[.41,.11,.48]
    elif headers==['物理意义','符号','数值'] and len(rows)==9: ratio=[.36,.17,.47]
    elif headers==['物理意义','符号','数值']: ratio=[.30,.24,.46]
    else:
        from statistics import median
        widths=[]
        for j in range(ncol):
            lengths=[weight(row.cells[j].text) for row in rows[1:]]
            widths.append(max(5.1, min(20, weight(headers[j]))*.60, min(21,median(lengths) if lengths else 6)))
        ratio=[w/sum(widths) for w in widths]
    sizes=[round(page_width*f) for f in ratio]; sizes[-1]+=page_width-sum(sizes)
    tab.autofit=False; tab.alignment=WD_TABLE_ALIGNMENT.CENTER
    pr=tab._tbl.tblPr
    pr.find(qn('w:tblW')).set(qn('w:type'),'dxa'); pr.find(qn('w:tblW')).set(qn('w:w'),str(page_width))
    for b in pr.xpath('./w:tblBorders|./w:tblLook|./w:tblCellMar'): b.getparent().remove(b)
    borders=OxmlElement('w:tblBorders'); pr.append(borders)
    for edge in ['top','bottom','left','right','insideV','insideH']:
        b=OxmlElement(f'w:{edge}'); b.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil'); b.set(qn('w:sz'),'8'); b.set(qn('w:color'),'202020'); borders.append(b)
    margins=OxmlElement('w:tblCellMar'); pr.append(margins)
    for edge,v in [('top','28'),('bottom','28'),('left','64'),('right','64')]:
        e=OxmlElement(f'w:{edge}'); e.set(qn('w:w'),v); e.set(qn('w:type'),'dxa'); margins.append(e)
    for col,w in zip(tab.columns,sizes): col.width=Pt(w/20)
    keep_short=(len(rows)<=9 or headers==['场景','方法','求解成功 / 1000','全链路实际达标 / 1000','决策耗时中位数 / s'])
    for j,row in enumerate(rows):
        rpr=row._tr.get_or_add_trPr(); rpr.append(OxmlElement('w:cantSplit'))
        if j==0 and rpr.find(qn('w:tblHeader')) is None: rpr.append(OxmlElement('w:tblHeader'))
        for k,(cell,w) in enumerate(zip(row.cells,sizes)):
            cell.width=Pt(w/20); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr()
            for elem in cp.xpath('./w:tcBorders|./w:shd'): elem.getparent().remove(elem)
            if j==0:
                b=OxmlElement('w:tcBorders'); bot=OxmlElement('w:bottom'); bot.set(qn('w:val'),'single'); bot.set(qn('w:sz'),'5'); b.append(bot); cp.append(b)
            for p in cell.paragraphs:
                p.style='Normal'; p.alignment=WD_ALIGN_PARAGRAPH.LEFT if k==0 else WD_ALIGN_PARAGRAPH.CENTER
                fmt=p.paragraph_format; fmt.space_before=Pt(0); fmt.space_after=Pt(0); fmt.line_spacing=1.06
                fmt.keep_with_next=j==0 or (keep_short and j<len(rows)-1); fmt.keep_together=True
                if j==0:
                    for run in p.runs: run.bold=True
from copy import deepcopy
from docx.text.run import Run
sup_count=0
for node in d.element.body.xpath('.//w:r[w:t]'):
    run=Run(node,d)
    t=run.text
    if 'SUPCITE' in t:
        parts=re.split(r'SUPCITE(\d+)ENDSUP',t)
        if len(parts)<3: continue
        for j,part in enumerate(parts):
            if not part: continue
            item=deepcopy(node); node.addprevious(item)
            rnew=Run(item,d); rnew.text=f'[{part}]' if j%2 else part
            if j%2:
                rnew.font.superscript=True; sup_count+=1
        node.getparent().remove(node)
assert sup_count==3, sup_count
# Format source-change markers and the algorithm frame.
pending=None
algorithm_started=False
for p in list(d.paragraphs):
    if p.text.startswith('CHANGEFLAG'):
        pending=p.text.removeprefix('CHANGEFLAG')
        p._p.getparent().remove(p._p)
        continue
    if pending:
        prefix_run=OxmlElement('w:r'); pr=OxmlElement('w:rPr')
        color=OxmlElement('w:color'); color.set(qn('w:val'),'174A85'); pr.append(color)
        prefix_run.append(pr); txt=OxmlElement('w:t');txt.text='['+pending+'] ';prefix_run.append(txt)
        p._p.insert(1 if p._p.pPr is not None else 0,prefix_run)
        for run in p.runs:run.font.color.rgb=RGBColor.from_string('174A85')
        pending=None
    if '算法1　预测驱动' in p.text:
        algorithm_started=True
        p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        b=OxmlElement('w:pBdr'); edge=OxmlElement('w:top');edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'8');b.append(edge);p._p.get_or_add_pPr().append(b)
    if algorithm_started:
        p.paragraph_format.keep_with_next=True
        p.paragraph_format.keep_together=True
        if p.text.strip()=='End':
            p.paragraph_format.keep_with_next=False
            b=OxmlElement('w:pBdr');edge=OxmlElement('w:bottom');edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'8');b.append(edge);p._p.get_or_add_pPr().append(b)
            algorithm_started=False

for sh in d.inline_shapes:
    # Use the existing raster at its native aspect ratio.
    width=Mm(160)
    if sh.width!=width:
        sh.height=round(sh.height*width/sh.width); sh.width=width

foot=sec.footer.paragraphs[0]; foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
foot.paragraph_format.space_after=Pt(0)
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); foot._p.append(field)
d.core_properties.title='不完美CSI下V2V RF/VLC异构通信的能效公平资源分配'
d.core_properties.subject='Paper3 中文全稿 v9 单栏审阅版'
d.core_properties.author=''
# Neutralize Pandoc's explicit fonts, retaining native Office mathematics.
for r in d.element.body.xpath('.//w:r'):
    rpr=r.find(qn('w:rPr'))
    if rpr is not None:
        fonts=rpr.find(qn('w:rFonts'))
        if fonts is not None:fonts.getparent().remove(fonts)
settings=d.settings.element
mathPr=settings.find(qn('m:mathPr'))
if mathPr is not None:
    mf=mathPr.find(qn('m:mathFont'))
    if mf is not None:mf.set(qn('m:val'),'Cambria Math')
d.save(output)
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    xml=z.read('word/document.xml').decode('utf-8')
    assert all(marker not in xml for marker in ['ALGORITHMLINEFLAG','EQNUMBERFLAG','SUPCITE','ENDSUP'])
    assert 'word/footnotes.xml' not in z.namelist() or '<w:footnote w:id="1">' not in z.read('word/footnotes.xml').decode()
    assert len(re.findall(r'<m:oMath(?: |>)',xml))>=300
print(json.dumps({'output':str(output),'tables':len(d.tables)-len(eq_tables),'images':len(d.inline_shapes),'native_equations':len(re.findall(r'<m:oMath(?: |>)',xml)),'numbered_displays':n_equations},ensure_ascii=False))
