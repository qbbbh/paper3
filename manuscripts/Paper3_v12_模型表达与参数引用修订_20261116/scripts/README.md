# 文档与图件生成

`build_word.py`从本包的完整Markdown生成清洁Word；加入`--annotated`生成修改标注版。使用Pandoc与python-docx。正文为小四宋体／Times New Roman，公式为原生Word数学对象。当前Windows机器的Pandoc位置在程序内明确给出；其他电脑应调整这一软件路径。

式（3）与附录式（A2）在Word中按等价的分行等式显示，以适配本机Word对分段公式大括号的渲染；Markdown保留分段形式。单位归一化和条件均保持。

`redraw_figures.py`读取本包data目录，导出六组图的PDF、SVG及PNG。需要NumPy、Matplotlib和实际Times New Roman字体。V12沿用V11图件，未重新运行科研求解器。

最终交付Word已经由Microsoft Word重新分页，并逐页视觉检查。重新生成文档后应重新渲染检查版面。
