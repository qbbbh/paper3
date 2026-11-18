"""Native Word layout for numbered mathematics and separately captioned panels."""
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import re

from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt
from docx.text.paragraph import Paragraph
from PIL import Image


def node(tag, **attrs):
    element = OxmlElement(tag)
    prefix = tag.split(":")[0]
    for key, value in attrs.items():
        element.set(qn(f"{prefix}:{key}"), str(value))
    return element


def extract_panels(markdown):
    panels = {}
    def extract(match):
        ident, body = match.groups()
        rows = [x.strip().strip("|").split("|") for x in body.strip().splitlines()]
        data = []
        for i, row in enumerate(rows):
            if not row[0].strip().startswith("!["):
                continue
            labels = rows[i+1] if i else rows[i+2]
            pair = []
            for col, label in zip(row, labels):
                path = re.fullmatch(r"!\[[^\]]+\]\(([^)]+)\)", col.strip()).group(1)
                pair.append((path, label.strip()))
            data.append(pair)
        assert len(data)==2 and all(len(r)==2 for r in data)
        panels[ident] = data
        return "REVIEWPANELS"+ident
    src = re.sub(r"<!-- panels (F[23]) -->\s*([\s\S]*?)\s*<!-- /panels -->", extract, markdown)
    assert set(panels)=={"F2","F3"}
    return src, panels


def embed_png(source, max_width_px):
    # Preserve the archival plots; only the embedded review copy is downsampled.
    with Image.open(source) as im:
        im = im.convert("RGB")
        if im.width > max_width_px:
            im = im.resize((max_width_px, round(im.height*max_width_px/im.width)), Image.Resampling.LANCZOS)
        out = BytesIO()
        im.quantize(colors=192, method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE).save(out,format="PNG",optimize=True,dpi=(240,240))
        return out.getvalue()


def put_panels(doc, base, panels):
    for p in list(doc.paragraphs):
        if not re.fullmatch(r"REVIEWPANELS(F2|F3)",p.text.strip()):
            continue
        ident=p.text.strip().removeprefix("REVIEWPANELS")
        rows=panels[ident]
        table=doc.add_table(rows=2*len(rows), cols=2)
        table.alignment=WD_TABLE_ALIGNMENT.CENTER
        table.autofit=False
        for col in table.columns:
            col.width=Mm(82)
        pr=table._tbl.tblPr
        pr.find(qn("w:tblW")).set(qn("w:type"),"dxa")
        pr.find(qn("w:tblW")).set(qn("w:w"),str(Mm(164).twips))
        borders=node("w:tblBorders")
        for key in ("top","left","bottom","right","insideH","insideV"):
            borders.append(node("w:"+key,val="nil"))
        pr.append(borders)
        mar=node("w:tblCellMar")
        for key,val in (("top",0),("left",20),("bottom",0),("right",20)):
            mar.append(node("w:"+key,w=val,type="dxa"))
        pr.append(mar)
        pr.append(node("w:tblCaption",val="Figure "+ident[1:]))
        for row in table.rows:
            row._tr.get_or_add_trPr().append(node("w:cantSplit"))
            for cell in row.cells:
                cell.width=Mm(82)
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                q=cell.paragraphs[0]
                q.style="Normal"; q.alignment=WD_ALIGN_PARAGRAPH.CENTER
                fmt=q.paragraph_format
                fmt.first_line_indent=Pt(0); fmt.left_indent=Pt(0); fmt.right_indent=Pt(0)
                fmt.space_after=Pt(0); fmt.space_before=Pt(0); fmt.line_spacing=1
        for rowno, pair in enumerate(rows):
            for col,(src,label) in enumerate(pair):
                image=table.cell(rowno*2,col).paragraphs[0]
                shape=image.add_run().add_picture(BytesIO(embed_png(base/src,775)),width=Mm(77.8))
                shape._inline.docPr.set("descr",label)
                shape._inline.docPr.set("name",Path(src).stem)
                cap=table.cell(rowno*2+1,col).paragraphs[0]
                cap.paragraph_format.space_before=Pt(1)
                cap.paragraph_format.space_after=Pt(4)
                cap.add_run(label)
        p._p.addprevious(table._tbl)
        p._p.getparent().remove(p._p)


def single_group(parent, number, rows, doc, width, next_id):
    # A true centred tab measures the whole native math object. A left-justified
    # single-column matrix aligns its rows internally without a visible separator.
    all_rows=[]
    for idx, row in enumerate(rows):
        content=[]
        for child in row:
            if child.tag==qn("m:argPr"): continue
            el=deepcopy(child)
            for text in el.iter(qn("m:t")):
                text.text=(text.text or "").replace("&","")
            for run in el.iter(qn("m:r")):
                pr=run.find(qn("w:rPr"))
                if pr is None:
                    pr=node("w:rPr");run.insert(1 if run.find(qn("m:rPr")) is not None else 0,pr)
                for key in ("sz","szCs"):
                    item=pr.find(qn("w:"+key))
                    if item is None:
                        pr.append(node("w:"+key,val=24))
                    else: item.set(qn("w:val"),"24")
            content.append(el)
        if number=="28":
            label="28"+chr(ord("a")+idx)
            last=[t for c in content for t in c.iter(qn("m:t"))][-1]
            assert last.text.endswith("("+label+")"),label
            last.text=last.text[:-len(label)-2]
        all_rows.append(content)
    grouped=[(number,all_rows)] if number!="28" else [
        ("28"+chr(ord("a")+i),[row]) for i,row in enumerate(all_rows)]
    for label,rowset in grouped:
        el=node("w:p")
        parent.addprevious(el)
        p=Paragraph(el,doc)
        p.style="Normal";p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        fmt=p.paragraph_format
        fmt.left_indent=Pt(0);fmt.right_indent=Pt(0);fmt.first_line_indent=Pt(0)
        fmt.space_before=Pt(3.4);fmt.space_after=Pt(3.6);fmt.line_spacing=1.08
        if number=="28":
            fmt.space_before=Pt(1);fmt.space_after=Pt(1)
        fmt.tab_stops.add_tab_stop(Pt(width/40),WD_TAB_ALIGNMENT.CENTER)
        fmt.tab_stops.add_tab_stop(Pt(width/20-.7),WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        mark=str(next_id[0]);next_id[0]+=1
        el.append(node("w:bookmarkStart",id=mark,name="PaperEq"+label))
        equation=node("m:oMath")
        if len(rowset)==1:
            equation.extend(rowset[0])
        else:
            matrix=node("m:m");prop=node("m:mPr")
            prop.extend([node("m:baseJc",val="center"),node("m:plcHide",val="1"),
                         node("m:rSpRule",val="0")])
            columns=node("m:mcs");column=node("m:mc");cp=node("m:mcPr")
            cp.extend([node("m:count",val="1"),node("m:mcJc",val="left")])
            column.append(cp);columns.append(column);prop.append(columns);matrix.append(prop)
            for children in rowset:
                r=node("m:mr");cell=node("m:e");cell.extend(children);r.append(cell);matrix.append(r)
            equation.append(matrix)
        el.append(equation);el.append(node("w:bookmarkEnd",id=mark))
        p.add_run("\t("+label+")")


def atom_style(run):
    prop=run.find(qn("m:rPr"))
    if run.find(qn("w:rPr")) is not None: return None
    style=None
    if prop is not None:
        if any(x.tag not in {qn("m:sty"),qn("m:nor")} for x in prop): return None
        sty=prop.find(qn("m:sty"))
        style=sty.get(qn("m:val")) if sty is not None else "p"
    text="".join(t.text or "" for t in run.iter(qn("m:t")))
    if not text or not re.fullmatch(r"[A-Za-z0-9Α-ωℓ]+",text):return None
    return (text, style)


def light_inline_symbols(doc):
    count=0
    for eq in list(doc.element.body.xpath(".//w:p/m:oMath")):
        p=eq.getparent()
        if any((b.get(qn("w:name")) or "").startswith("PaperEq") for b in p.findall(qn("w:bookmarkStart"))):
            continue
        if len(eq)!=1: continue
        atom=eq[0]
        if atom.tag==qn("m:r"):
            seq=[(atom,None)]
        elif atom.tag in (qn("m:sSub"),qn("m:sSup")):
            role="sub" if atom.tag==qn("m:sSub") else "sup"
            base,script=atom.find(qn("m:e")),atom.find(qn("m:"+role))
            if base is None or script is None or len(base)!=1 or len(script)!=1:continue
            seq=[(base[0],None),(script[0],role)]
        else:continue
        if any(x.tag!=qn("m:r") or atom_style(x) is None for x,_ in seq):continue
        for item,position in seq:
            text,style=atom_style(item)
            run=node("w:r");pr=node("w:rPr")
            pr.append(node("w:rFonts",ascii="Times New Roman",hAnsi="Times New Roman",eastAsia="Times New Roman",cs="Times New Roman"))
            if style in ("b","bi"):pr.append(node("w:b"))
            if (style in ("i","bi") or style is None) and not text.isdecimal():pr.append(node("w:i"))
            pr.append(node("w:noProof"))
            if position: pr.append(node("w:vertAlign",val="subscript" if position=="sub" else "superscript"))
            run.append(pr);lit=node("w:t");lit.text=text;run.append(lit);eq.addprevious(run)
        p.remove(eq);count+=1
    return count


def prepare_review(doc):
    # Remove the source of the pagination squares, including inherited styles.
    # Widow/orphan control can still keep a single line off a new page.
    flags=("keepNext","keepLines","pageBreakBefore","suppressLineNumbers")
    for root in (doc.element,doc.styles.element,doc.sections[0].footer._element):
        for flag in flags:
            for el in list(root.iter(qn("w:"+flag))):
                if root is doc.styles.element:
                    el.set(qn("w:val"),"0")
                else:
                    el.getparent().remove(el)
    for st in doc.styles:
        if st.style_id.startswith("Heading") or st.style_id in ("Title","Caption"):
            for name in ("keep_with_next","keep_together","page_break_before"):
                setattr(st.paragraph_format,name,False) if hasattr(st,"paragraph_format") else None
    settings=doc.settings.element
    compat=settings.find(qn("w:compat"))
    if compat is None:
        compat=node("w:compat")
        settings.insert_element_before(compat,"w:rsids","m:mathPr","w:themeFontLang","w:clrSchemeMapping")
    targets=[r for r in compat.findall(qn("w:compatSetting")) if r.get(qn("w:name"))=="compatibilityMode"]
    if targets: targets[0].set(qn("w:val"),"15")
    else: compat.append(node("w:compatSetting",name="compatibilityMode",uri="http://schemas.microsoft.com/office/word",val="15"))
    for tag in ("embedSystemFonts","savePreviewPicture"):
        for el in list(settings.findall(qn("w:"+tag))):settings.remove(el)
    for lang in doc.styles.element.xpath("./w:docDefaults/w:rPrDefault/w:rPr/w:lang"):
        lang.set(qn("w:eastAsia"),"zh-CN")
    return light_inline_symbols(doc)
