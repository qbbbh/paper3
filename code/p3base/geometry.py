"""几何工具：端口坐标、建筑棱柱遮挡、车辆矩形遮挡。

全部使用布尔数组；旧代码里的 uint16 遮挡位掩码在 50/100 车下会溢出，此处不再使用。
坐标统一为场景局部米制、z 轴向上、地面 z=0。
"""
import numpy as np


def unit_from_heading(heading_rad):
    heading_rad = np.asarray(heading_rad, float)
    return np.stack([np.cos(heading_rad), np.sin(heading_rad)], axis=-1)


def vehicle_boxes(center_xy, heading_rad, length_m, width_m, height_m):
    """车辆定向矩形的轴对齐包围盒，只用于粗筛；精确遮挡用 blocked_by_vehicles。"""
    c = np.asarray(center_xy, float)
    half_diag = 0.5 * np.hypot(length_m, width_m)
    boxes = np.empty((len(c), 6))
    boxes[:, 0:2] = c - half_diag
    boxes[:, 2] = 0.0
    boxes[:, 3:5] = c + half_diag
    boxes[:, 5] = height_m
    return boxes


def segments_hit_boxes(start, end, boxes):
    """逐元素线段-长方体求交；boxes 形状为 start.shape[:-1] + (6,)。

    盒内端点为遮挡；退化线段（平行且贴面）按包容处理。
    """
    start, end = np.broadcast_arrays(start, end)
    direction = end - start
    shape = start.shape[:-1]
    enter = np.zeros(shape)
    leave = np.ones(shape)
    possible = np.ones(shape, dtype=bool)
    for axis in range(3):
        d = direction[..., axis]
        x = start[..., axis]
        lo = boxes[..., axis]
        hi = boxes[..., axis + 3]
        parallel = np.abs(d) < 1e-12
        possible &= ~(parallel & ((x < lo) | (x > hi)))
        denom = np.where(parallel, 1.0, d)
        t1 = (lo - x) / denom
        t2 = (hi - x) / denom
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(t1, t2)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(t1, t2)))
    return possible & (enter <= leave)


def blocked_segments(start, end, boxes):
    """一组共享建筑盒对多条线段求交；boxes 形状 (m,6)，返回 start.shape[:-1]。"""
    start, end = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
    out = np.zeros(start.shape[:-1], dtype=bool)
    target = start.shape[:-1] + (6,)
    for box in np.asarray(boxes, float).reshape(-1, 6):
        out |= segments_hit_boxes(start, end, np.broadcast_to(box, target))
    return out


def blocked_by_vehicles(start, end, centers, headings, length_m, width_m, height_m):
    """车辆定向矩形遮挡。

    start/end 形状 (..., 3)，centers (m, 2)，headings (m,)。把端点变换到每辆遮挡车
    的局部坐标（x 沿车头、y 沿车宽），再与 [-L/2,L/2]×[-W/2,W/2]×[0,H] 求交。
    返回形状 (..., m)，调用方按需 any(axis=-1)。
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    centers = np.asarray(centers, float)
    headings = np.asarray(headings, float)
    m = len(centers)
    if m == 0:
        return np.zeros(start.shape[:-1] + (0,), dtype=bool)
    u = unit_from_heading(headings)                      # (m,2)
    normal = np.stack([-u[:, 1], u[:, 0]], axis=-1)      # 左法线

    def local(points):
        rel = points[..., None, :2] - centers             # (..., m, 2)
        xy = np.stack([(rel * u).sum(-1), (rel * normal).sum(-1)], axis=-1)
        z = np.broadcast_to(points[..., None, 2:3], xy.shape[:-1] + (1,))
        return np.concatenate([xy, z], axis=-1)

    half_l = 0.5 * np.broadcast_to(np.asarray(length_m, float), (m,))
    half_w = 0.5 * np.broadcast_to(np.asarray(width_m, float), (m,))
    height = np.broadcast_to(np.asarray(height_m, float), (m,))
    boxes = np.stack([
        -half_l, -half_w, np.zeros(m), half_l, half_w, height,
    ], axis=1)
    return segments_hit_boxes(local(start), local(end), boxes)


def emission_incidence(port_xy, outward_dir, peer_xy, peer_dir):
    """发射角与入射角余弦。

    cos_tx：LED 外法线与传播方向同向，要求 >0（前向半空间）。
    cos_rx：PD 外法线与"指向对端"的方向同向，要求 >= cos(FOV)。
    返回 (端口距离, cos_tx, cos_rx)。
    """
    v = peer_xy - port_xy
    d = np.linalg.norm(v, axis=-1)
    safe = np.where(d > 1e-12, d, 1.0)
    vhat = v / safe[..., None]
    cos_tx = np.sum(outward_dir * vhat, axis=-1)
    cos_rx = np.sum(peer_dir * (-vhat), axis=-1)
    return d, np.clip(cos_tx, -1.0, 1.0), np.clip(cos_rx, -1.0, 1.0)
