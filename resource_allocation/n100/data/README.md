# 100车场景：NPZ计算原件与CSV阅读版本

1001帧，评价时间0–100 s，原生步长0.1 s，已排除30 s预热。共 **100,100 条车辆—时刻记录**。随机种子和道路参数见 [scenario.json](scenario.json)，交通检查见 [audit.json](audit.json)。这是一条开发轨迹，不是多条独立交通轨迹。

## Chat从哪里开始

先读 [两帧预览](csv/mobility_preview.csv)、[车辆尺寸](csv/vehicles.csv)、[建筑](csv/buildings.csv)。全部记录保存在csv目录的 `mobility_rows_*.csv`，按行范围切分；没有抽样丢弃中间时间。字段与来源摘要见 [csv_export_manifest.json](csv_export_manifest.json)。

|字段|含义和单位|
|---|---|
|frame_index、vehicle_index|从0开始的帧/车辆索引|
|vehicle_id|SUMO车辆标识，不能当数值排序替代索引|
|time_s|评价段相对时间，秒|
|simulation_time_s|SUMO包含预热的时间，秒|
|front_x_m/front_y_m、center_x_m/center_y_m、rear_x_m/rear_y_m|前/中心/后部位置，局部米制坐标|
|heading_rad|朝向，弧度；物理层方向为cos/sin。CSV保留原值，未unwrap|
|speed_mps、acceleration_mps2|速度m/s、加速度m/s²|
|lane_id、lane_position_m|SUMO车道标识与车道位置|
|vehicle_length_m/width_m/height_m|车辆尺寸，放在vehicles.csv|
|buildings.csv的min/max xyz|建筑轴对齐盒，单位米；不是单独的信道增益|

CSV采用可往返的浮点文本，逐字段还原后与NPZ的全部数组完全一致，结果记录于manifest。它供阅读核对，仿真仍读取原始NPZ。CSV并不保证连接器会一次读取全部行，应按块/车检索。

`mobility.npz`和`buildings.npz`是原件；若克隆后缺少大文件，先运行仓库根目录 `python tools/run.py prepare` 从data_parts无损还原。NPZ无需SUMO即可运行已有轨迹，但不能据此说仿真包含真实包接收过程。
