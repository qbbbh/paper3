# 重绘与Word生成

`redraw_figures.py` 读取 ../data 中冻结的CSV，输出 ../figures。依赖 numpy、matplotlib 和本机 Times New Roman；不调用科研求解器。

`build_word.py` 从清洁Markdown生成Word；加 `--annotated` 生成标注Word。依赖 python-docx 与 Pandoc。Windows默认Pandoc位置为 C:/Program Files/Pandoc/pandoc.exe；其他安装位置需修改脚本中的路径。中间文件写入系统临时目录。生成后仍需用Word打开、分页并检查；随包PDF是本次已检查的布局快照。

本包没有重新执行主实验或需求敏感性实验。
