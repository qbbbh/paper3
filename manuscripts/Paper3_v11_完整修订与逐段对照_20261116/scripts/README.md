# 文稿与图件重建

## 图件

`python redraw_figures.py`读取`../data/`的冻结表格，生成六组PDF/SVG/PNG。需要NumPy、Matplotlib及实际安装的Times New Roman；本次输出使用Windows系统的`times.ttf`。字体与数据摘要在`figures/figure_audit.json`。

## Word

`python build_word.py`生成清洁版；`python build_word.py --annotated`生成修改标注版。需要Python、python-docx及Windows上的Pandoc（脚本指定`C:\Program Files\Pandoc\pandoc.exe`）。输入为包根目录的两份Markdown，使用包内PNG图件，数学表达转换为Word原生OMML。

正文及表格为12 pt宋体／Times New Roman，公式使用Word默认Cambria Math；A4单栏。最终交付DOCX另外经过本机Microsoft Word打开、重排并导出PDF，页数以该PDF为准。不同Word版本的分页可能略有差别。
