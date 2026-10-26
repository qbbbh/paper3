"""场景读取层：10/50/100 车共用，不限定车辆数。

数据来自 SUMO_10_50_100车_建筑公园_20261023/n{10,50,100}/：
  mobility.npz  逐 100 ms 记录的位置/朝向/速度，原生步长 0.1 s
  buildings.npz 建筑轴对齐棱柱 boxes_xyz_m，形状 (m,6)

时间语义：t=0 为 100 s 评价段起点（30 s 预热已排除）。块 k 的时间为 k*block_dt。
at(t) 使用线性插值，仅供物理评价；决策器只能使用 knot(index)，不得读取未来端点。
"""
from pathlib import Path

import numpy as np

from .config import VehicleConfig
from .geometry import unit_from_heading


class Scene:
    def __init__(self, directory, vehicle: VehicleConfig | None = None):
        self.directory = Path(directory)
        self.vehicle = vehicle or VehicleConfig()
        with np.load(self.directory / "mobility.npz", allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        with np.load(self.directory / "buildings.npz", allow_pickle=False) as z:
            self.buildings = z["boxes_xyz_m"].astype(float)
        self.time_s = self.data["time_s"].astype(float)
        self.center_xy = self.data["center_xy_m"].astype(float)
        self.heading_rad = np.unwrap(self.data["heading_rad"].astype(float), axis=0)
        self.speed_mps = self.data["speed_mps"].astype(float)
        self.length_m = self.data["vehicle_length_m"].astype(float)
        self.width_m = self.data["vehicle_width_m"].astype(float)
        self.height_m = self.data["vehicle_height_m"].astype(float)
        self.vehicle_ids = self.data["vehicle_ids"]
        if self.center_xy.ndim != 3 or self.center_xy.shape[-1] != 2:
            raise ValueError("mobility.npz 的 center_xy_m 形状应为 (帧, 车, 2)")
        self.n = self.center_xy.shape[1]
        self.frame_dt = float(np.median(np.diff(self.time_s)))
        if not np.isclose(self.frame_dt, 0.1, atol=1e-6):
            raise ValueError(f"mobility 原生步长应为 0.1 s，实际 {self.frame_dt}")

    # ---- 时间与索引 ----------------------------------------------------
    def block_count(self, block_dt=0.1):
        span = self.time_s[-1] - self.time_s[0]
        return int(np.floor(span / block_dt + 1e-9))

    def knot(self, index):
        """只读一个已记录帧；因果决策器应使用这个接口。"""
        i = int(index)
        return dict(
            center_xy_m=self.center_xy[i].copy(),
            heading_rad=self.heading_rad[i].copy(),
            speed_mps=self.speed_mps[i].copy(),
        )

    def _bracket(self, t):
        times = self.time_s
        t = float(t)
        if not times[0] - 1e-9 <= t <= times[-1] + 1e-9:
            raise ValueError(f"时间 {t} 超出记录区间 [{times[0]}, {times[-1]}]")
        hi = int(min(np.searchsorted(times, t, side="right"), len(times) - 1))
        lo = max(hi - 1, 0)
        span = times[hi] - times[lo]
        w = 0.0 if span <= 0 else (t - times[lo]) / span
        return lo, hi, w

    def at(self, t):
        """线性插值几何；物理评价用，不代表 SUMO 的 1 ms 原生采样。"""
        lo, hi, w = self._bracket(t)
        return dict(
            center_xy_m=(1 - w) * self.center_xy[lo] + w * self.center_xy[hi],
            heading_rad=(1 - w) * self.heading_rad[lo] + w * self.heading_rad[hi],
            speed_mps=(1 - w) * self.speed_mps[lo] + w * self.speed_mps[hi],
        )

    # ---- 端口 ----------------------------------------------------------
    def ports(self, state=None, t=None):
        """返回前/后保险杠端口。

        pos: (n, 2, 3)，第 1 维 0=前端口、1=后端口，z 为 port_height_m
        outward: (n, 2, 2)，前端口朝 +heading，后端口朝 -heading
        """
        if state is None:
            if t is None:
                raise ValueError("需要 state 或 t")
            state = self.at(t)
        c = np.asarray(state["center_xy_m"], float)
        h = np.asarray(state["heading_rad"], float)
        u = unit_from_heading(h)
        half = 0.5 * self.length_m[:, None]
        planar = np.stack([c + half * u, c - half * u], axis=1)       # (n,2,2)
        z = np.full(planar.shape[:-1] + (1,), self.vehicle.port_height_m)
        pos = np.concatenate([planar, z], axis=-1)                    # (n,2,3)
        outward = np.stack([u, -u], axis=1)                           # (n,2,2)
        return dict(pos=pos, outward=outward, center_xy_m=c,
                    heading_rad=h, speed_mps=np.asarray(state["speed_mps"], float))

    def vehicle_geometry(self):
        return dict(length_m=self.length_m, width_m=self.width_m,
                    height_m=self.height_m)

    # ---- 元信息 --------------------------------------------------------
    def summary(self):
        return dict(vehicles=int(self.n), frames=int(len(self.time_s)),
                    frame_dt_s=self.frame_dt,
                    duration_s=float(self.time_s[-1] - self.time_s[0]),
                    buildings=int(len(self.buildings)),
                    directory=str(self.directory))


def find_scene_root(code_root: Path, n: int) -> Path:
    """按车辆数定位场景目录，避免任何 6/8/10 的硬编码分支。"""
    root = Path(code_root) / "SUMO_10_50_100车_建筑公园_20261023"
    directory = root / f"n{int(n)}"
    if not directory.is_dir():
        raise FileNotFoundError(f"未找到 {n} 车场景：{directory}")
    return directory
