# 无损分段数据

连接器的大请求传输失败，因此只将超过800000字节的文件分段保存。运行 `python tools/restore_data.py` 还原原路径，再运行 `python tools/verify_snapshot.py`。每段和完整文件均验证SHA-256；不修改任何数据数值。CSV与摘要仍可直接在线阅读。
