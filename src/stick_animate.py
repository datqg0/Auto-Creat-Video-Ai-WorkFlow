"""Module Hoạt hoạ Người que (Stick Figure Animation Engine) lấy cảm hứng từ StykAnimate.

Tích hợp các thuật toán cốt lõi từ StykAnimate:
1. 2D Inverse Kinematics (IK):
   - Analytic 2-Bone IK (Law of Cosines với điều khiển hướng gập elbow/knee).
   - FABRIK (Forward And Backward Reaching Inverse Kinematics) cho chuỗi n-xương.
   - Pinned IK targets (giữ cố định chân trên mặt đất khi hông dao động).
2. Bone Rigging & Hierarchy chuẩn Pivot/StykAnimate (Torso, Head, Shoulders, Arms, Hips, Legs).
3. Thư viện Procedural Poses & Actions:
   - idle (thở, nhấp nhô tự nhiên)
   - walk (chu kỳ bước đi với chân cắm mốc IK, hông nhịp nhàng)
   - run (chạy tốc độ cao, thân nghiêng)
   - pointing (tay chỉ deictic vào biểu thức toán / sơ đồ / code với que chỉ laser neon)
   - confused (gãi đầu, nghiêng đầu, dấu hỏi trôi nổi)
   - idea (bật nhảy, giơ cao hai tay, hào quang & bóng đèn phát sáng)
   - coding (gõ bàn phím trên laptop holographic phát sáng)
   - celebrate (ăn mừng thành công)
   - struggle (đẩy khối nặng / vật cản)
4. Hiệu ứng hạt VFX (Aura neon, Shockwave, Tia sáng).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 1. Thuật toán Inverse Kinematics (IK)
# ---------------------------------------------------------------------------

def solve_ik_2bone(
    base: Point,
    target: Point,
    l1: float,
    l2: float,
    bend_dir: int = 1,
    allow_stretch: bool = False,
) -> Tuple[Point, Point]:
    """Giải tích 2-bone IK thuần túy (Analytic Two-Bone Inverse Kinematics).

    Dựa trên định lý hàm Cosin (Law of Cosines), giải tức thời vị trí khớp giữa (elbow/knee)
    và khớp cuối (hand/foot).
    - base: Toạ độ khớp gốc (vai hoặc hông).
    - target: Toạ độ đích mong muốn (bàn tay hoặc bàn chân).
    - l1, l2: Chiều dài đoạn xương 1 và đoạn xương 2.
    - bend_dir: Hướng gập (+1 hoặc -1).
    - allow_stretch: Nếu True, tự động kéo giãn đều khi mục tiêu vượt tầm với.

    Trả về: (mid_joint, end_joint)
    """
    bx, by = base
    tx, ty = target
    dx = tx - bx
    dy = ty - by
    dist = math.hypot(dx, dy)
    eps = 1e-6

    total_len = l1 + l2

    if dist < eps:
        # Trường hợp trùng điểm gốc: gập xương vuông góc theo bend_dir
        angle = math.pi / 2 * bend_dir
        mx = bx + math.cos(angle) * l1
        my = by + math.sin(angle) * l1
        return (mx, my), (bx, by)

    ux = dx / dist
    uy = dy / dist

    # Xử lý khi vượt tầm với (unreachable target)
    if dist >= total_len - eps:
        if allow_stretch:
            ratio = dist / max(total_len, eps)
            al1 = l1 * ratio
            return (bx + ux * al1, by + uy * al1), (tx, ty)
        else:
            return (bx + ux * l1, by + uy * l1), (bx + ux * total_len, by + uy * total_len)

    # Đảm bảo khoảng cách không nhỏ hơn độ chênh lệch 2 xương
    min_dist = abs(l1 - l2)
    d = max(dist, min_dist + eps)

    # Chiều dài hình chiếu lên trục base -> target
    along = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    height_sq = l1 * l1 - along * along
    height = math.sqrt(max(0.0, height_sq)) * bend_dir

    # Toạ độ khớp giữa (khuỷu tay hoặc đầu gối)
    mx = bx + ux * along - uy * height
    my = by + uy * along + ux * height

    # Toạ độ khớp cuối đạt tới target
    ex = bx + ux * d
    ey = by + uy * d

    return (mx, my), (ex, ey)


def solve_ik_fabrik(
    points: List[Point],
    lengths: List[float],
    target: Point,
    iterations: int = 16,
    tolerance: float = 1e-3,
) -> List[Point]:
    """Thuật toán FABRIK (Forward And Backward Reaching Inverse Kinematics) từ StykAnimate.

    Dùng cho các chuỗi nhiều hơn 2 khớp (như sống lưng mềm, đuôi, xúc tu, hoặc cánh tay nhiều khớp).
    """
    n = len(lengths)
    if n == 0 or len(points) != n + 1:
        return list(points)

    pts = [list(p) for p in points]
    base = list(pts[0])
    total_len = sum(lengths)

    tx, ty = target
    dist_to_target = math.hypot(tx - base[0], ty - base[1])

    # Nếu ngoài tầm với, duỗi thẳng về hướng target
    if dist_to_target >= total_len:
        if dist_to_target < 1e-6:
            return points
        ux = (tx - base[0]) / dist_to_target
        uy = (ty - base[1]) / dist_to_target
        res = [tuple(base)]
        cur_x, cur_y = base[0], base[1]
        for l in lengths:
            cur_x += ux * l
            cur_y += uy * l
            res.append((cur_x, cur_y))
        return res

    for _ in range(iterations):
        # 1. Forward pass (từ ngọn về gốc)
        pts[-1] = [tx, ty]
        for i in range(n - 1, -1, -1):
            cur = pts[i + 1]
            prev = pts[i]
            d = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            if d > 1e-6:
                r = lengths[i] / d
                pts[i] = [cur[0] + (prev[0] - cur[0]) * r, cur[1] + (prev[1] - cur[1]) * r]

        # 2. Backward pass (từ gốc ra ngọn)
        pts[0] = list(base)
        for i in range(n):
            cur = pts[i]
            nxt = pts[i + 1]
            d = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if d > 1e-6:
                r = lengths[i] / d
                pts[i + 1] = [cur[0] + (nxt[0] - cur[0]) * r, cur[1] + (nxt[1] - cur[1]) * r]

        err = math.hypot(pts[-1][0] - tx, pts[-1][1] - ty)
        if err < tolerance:
            break

    return [tuple(p) for p in pts]


# ---------------------------------------------------------------------------
# 2. Bộ khung Người que (Stick Figure Rig)
# ---------------------------------------------------------------------------

@dataclass
class StickmanPoseData:
    """Toạ độ các khớp xương của Stick Figure đã tính toán."""
    pelvis: Point
    neck: Point
    head: Point
    head_radius: float

    left_shoulder: Point
    left_elbow: Point
    left_hand: Point

    right_shoulder: Point
    right_elbow: Point
    right_hand: Point

    left_hip: Point
    left_knee: Point
    left_foot: Point

    right_hip: Point
    right_knee: Point
    right_foot: Point

    # Thuộc tính phụ trợ cho diễn hoạt
    prop_type: str = ""       # "laptop", "laser_pointer", "bulb", "question", "sword"
    prop_pos: Point = (0, 0)
    aura_intensity: float = 0.0
    aura_color: str = "#00f0ff"
    eye_offset: Point = (0, 0)


class StickmanRig:
    """Bộ khung rigging người que tỉ lệ chuẩn Pivot/StykAnimate.

    Kích thước chuẩn (tỉ lệ 1.0):
    - Head radius: 20
    - Torso length: 65
    - Upper arm / Forearm: 32 / 30
    - Thigh / Shin: 38 / 36
    - Shoulder width: 28
    - Hip width: 20
    """

    def __init__(
        self,
        head_radius: float = 20.0,
        torso_len: float = 65.0,
        upper_arm_len: float = 32.0,
        forearm_len: float = 30.0,
        thigh_len: float = 38.0,
        shin_len: float = 36.0,
        shoulder_width: float = 28.0,
        hip_width: float = 20.0,
    ) -> None:
        self.head_radius = head_radius
        self.torso_len = torso_len
        self.upper_arm_len = upper_arm_len
        self.forearm_len = forearm_len
        self.thigh_len = thigh_len
        self.shin_len = shin_len
        self.shoulder_width = shoulder_width
        self.hip_width = hip_width

    def evaluate_pose(
        self,
        ground_x: float,
        ground_y: float,
        pose_name: str = "idle",
        t: float = 0.0,
        scale: float = 1.0,
        target_point: Optional[Point] = None,
        color: str = "#00f0ff",
    ) -> StickmanPoseData:
        """Đánh giá toàn bộ toạ độ xương của Người Que tại thời điểm t theo pose yêu cầu."""
        s = scale
        hr = self.head_radius * s
        tl = self.torso_len * s
        ual = self.upper_arm_len * s
        fal = self.forearm_len * s
        thl = self.thigh_len * s
        shl = self.shin_len * s
        sw = self.shoulder_width * s * 0.5
        hw = self.hip_width * s * 0.5

        leg_total = thl + shl
        # Chiều cao đứng tiêu chuẩn
        stand_pelvis_y = ground_y - leg_total * 0.96

        # Chọn pose tương ứng
        pname = pose_name.lower().strip()
        if "walk" in pname:
            return self._pose_walk(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "run" in pname:
            return self._pose_run(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "point" in pname or "explain" in pname or "present" in pname:
            return self._pose_pointing(ground_x, stand_pelvis_y, ground_y, s, t, target_point, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "confused" in pname or "problem" in pname or "doubt" in pname:
            return self._pose_confused(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "idea" in pname or "solution" in pname or "aha" in pname or "bulb" in pname:
            return self._pose_idea(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "code" in pname or "typing" in pname or "laptop" in pname or "hacker" in pname:
            return self._pose_coding(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "celebrate" in pname or "victory" in pname or "win" in pname:
            return self._pose_celebrate(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        elif "push" in pname or "struggle" in pname:
            return self._pose_pushing(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)
        else:
            return self._pose_idle(ground_x, stand_pelvis_y, ground_y, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color)

    # -----------------------------------------------------------------------
    # Triển khai từng Procedural Pose sử dụng IK
    # -----------------------------------------------------------------------

    def _pose_idle(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose đứng thở nhịp nhàng, tay buông tự nhiên."""
        breath = math.sin(t * 3.0) * (3.0 * s)
        pelvis = (gx, py + breath)
        neck = (gx, pelvis[1] - tl)
        head = (gx, neck[1] - hr * 1.1)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)

        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Chân đứng thẳng vững trên mặt đất (Pinned IK)
        l_foot = (l_hip[0] - 6 * s, gy)
        r_foot = (r_hip[0] + 6 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Tay hơi đung đưa nhẹ
        arm_sw = math.sin(t * 2.5) * (4 * s)
        l_hand = (l_sh[0] - 12 * s, l_sh[1] + (ual + fal) * 0.88 + arm_sw)
        r_hand = (r_sh[0] + 12 * s, r_sh[1] + (ual + fal) * 0.88 - arm_sw)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            eye_offset=(2 * s, 0),
        )

    def _pose_walk(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose chu kỳ bước đi (Walk Cycle) với bàn chân cắm mốc IK (Plant & Push)."""
        walk_speed = 5.0
        phase = t * walk_speed
        stride = 24.0 * s
        lift = 14.0 * s

        # Hông nhấp nhô theo hình sin kép (2 bước chân / chu kỳ)
        hip_bob = abs(math.sin(phase)) * (5.0 * s)
        pelvis = (gx, py - (4 * s) + hip_bob)
        torso_tilt = 4.0 * s  # hơi nghiêng người về phía trước
        neck = (gx + torso_tilt, pelvis[1] - tl)
        head = (neck[0] + 2 * s, neck[1] - hr * 1.1)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Bàn chân di chuyển theo chu kỳ elip (nửa chu kỳ cắm đất lướt sau, nửa chu kỳ nhấc lên bước tới)
        sin_l = math.sin(phase)
        cos_l = math.cos(phase)
        l_foot_x = l_hip[0] + sin_l * stride
        l_foot_y = gy - max(0.0, -cos_l) * lift

        sin_r = math.sin(phase + math.pi)
        cos_r = math.cos(phase + math.pi)
        r_foot_x = r_hip[0] + sin_r * stride
        r_foot_y = gy - max(0.0, -cos_r) * lift

        (l_knee, _) = solve_ik_2bone(l_hip, (l_foot_x, l_foot_y), thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, (r_foot_x, r_foot_y), thl, shl, bend_dir=1)

        # Tay đánh ngược pha với chân (cơ sinh học đối xứng)
        l_hand = (l_sh[0] - sin_l * (stride * 0.8), l_sh[1] + (ual + fal) * 0.8)
        r_hand = (r_sh[0] - sin_r * (stride * 0.8), r_sh[1] + (ual + fal) * 0.8)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=(l_foot_x, l_foot_y),
            right_hip=r_hip, right_knee=r_knee, right_foot=(r_foot_x, r_foot_y),
            eye_offset=(4 * s, 0),
        )

    def _pose_run(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose chạy nhanh (Run Cycle) năng động."""
        run_speed = 8.5
        phase = t * run_speed
        stride = 36.0 * s
        lift = 20.0 * s

        hip_bob = abs(math.sin(phase)) * (8.0 * s)
        pelvis = (gx, py - (8 * s) + hip_bob)
        torso_tilt = 12.0 * s  # nghiêng mạnh về phía trước
        neck = (gx + torso_tilt, pelvis[1] - tl * 0.94)
        head = (neck[0] + 6 * s, neck[1] - hr * 1.05)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        sin_l = math.sin(phase)
        cos_l = math.cos(phase)
        l_foot_x = l_hip[0] + sin_l * stride
        l_foot_y = gy - max(0.0, -cos_l) * lift

        sin_r = math.sin(phase + math.pi)
        cos_r = math.cos(phase + math.pi)
        r_foot_x = r_hip[0] + sin_r * stride
        r_foot_y = gy - max(0.0, -cos_r) * lift

        (l_knee, _) = solve_ik_2bone(l_hip, (l_foot_x, l_foot_y), thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, (r_foot_x, r_foot_y), thl, shl, bend_dir=1)

        # Tay gập vuông góc chạy mạnh mẽ
        l_hand = (l_sh[0] - sin_l * (stride * 0.9), l_sh[1] + ual * 0.5 - cos_l * 10 * s)
        r_hand = (r_sh[0] - sin_r * (stride * 0.9), r_sh[1] + ual * 0.5 - cos_r * 10 * s)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=(l_foot_x, l_foot_y),
            right_hip=r_hip, right_knee=r_knee, right_foot=(r_foot_x, r_foot_y),
            eye_offset=(6 * s, 1 * s),
        )

    def _pose_pointing(self, gx, py, gy, s, t, target_point, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose chỉ tay deictic vào bảng/công thức/code với que laser neon."""
        breath = math.sin(t * 3.0) * (2.0 * s)
        pelvis = (gx, py + breath)
        neck = (gx + 3 * s, pelvis[1] - tl)
        head = (gx + 4 * s, neck[1] - hr * 1.1)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Chân đứng trụ vững vàng
        l_foot = (l_hip[0] - 10 * s, gy)
        r_foot = (r_hip[0] + 14 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Tay trái chống hông tự tin
        l_hand = (l_hip[0] - 8 * s, l_hip[1] - 4 * s)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)

        # Tay phải dùng IK vươn chỉ về hướng target_point (hoặc điểm mặc định bên phải)
        tx = target_point[0] if target_point else gx + 70 * s
        ty = target_point[1] if target_point else neck[1] - 15 * s

        # Cho phép tay vươn tối đa theo hướng đó
        reach_dir_x = tx - r_sh[0]
        reach_dir_y = ty - r_sh[1]
        reach_dist = math.hypot(reach_dir_x, reach_dir_y)
        max_reach = (ual + fal) * 0.95
        if reach_dist > max_reach and reach_dist > 1e-6:
            r_hand_x = r_sh[0] + (reach_dir_x / reach_dist) * max_reach
            r_hand_y = r_sh[1] + (reach_dir_y / reach_dist) * max_reach
        else:
            r_hand_x = tx
            r_hand_y = ty

        r_hand = (r_hand_x, r_hand_y)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            prop_type="laser_pointer",
            prop_pos=r_hand,
            eye_offset=(5 * s, -1 * s),
        )

    def _pose_confused(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose bối rối: nghiêng đầu, một tay gãi đầu, dấu hỏi phát sáng trên đầu."""
        sway = math.sin(t * 2.5) * (4.0 * s)
        head_tilt = math.sin(t * 3.0) * (3.0 * s)

        pelvis = (gx + sway * 0.5, py)
        neck = (gx + sway, pelvis[1] - tl)
        # Đầu nghiêng hẳn một bên
        head = (neck[0] + 8 * s + head_tilt, neck[1] - hr * 1.05)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        l_foot = (l_hip[0] - 12 * s, gy)
        r_foot = (r_hip[0] + 8 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Tay phải đưa lên gãi sau đầu với IK
        scratch_wobble = math.sin(t * 8.0) * (3.0 * s)
        r_hand = (head[0] + hr * 0.8, head[1] - hr * 0.3 + scratch_wobble)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        # Tay trái buông xệ hoặc xoè bối rối
        l_hand = (l_sh[0] - 18 * s, l_sh[1] + (ual + fal) * 0.7)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            prop_type="question",
            prop_pos=(head[0], head[1] - hr * 1.6),
            eye_offset=(1 * s, -3 * s),
        )

    def _pose_idea(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose bừng sáng ý tưởng (Aha moment): nhảy cẫng lên, hai tay vươn cao, bóng đèn phát sáng."""
        jump = -abs(math.sin(t * 6.0)) * (16.0 * s)
        pelvis = (gx, py + jump)
        neck = (gx, pelvis[1] - tl)
        head = (gx, neck[1] - hr * 1.1)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Chân co nhẹ khi nhảy
        l_foot = (l_hip[0] - 8 * s, gy + jump * 0.5)
        r_foot = (r_hip[0] + 8 * s, gy + jump * 0.5)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Hai tay vươn lên mừng rỡ chữ V
        l_hand = (l_sh[0] - 26 * s, neck[1] - 32 * s)
        r_hand = (r_sh[0] + 26 * s, neck[1] - 32 * s)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            prop_type="bulb",
            prop_pos=(head[0], head[1] - hr * 1.8),
            aura_intensity=1.0,
            aura_color="#ffe600",
            eye_offset=(0, -2 * s),
        )

    def _pose_coding(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose ngồi làm việc / lập trình viên gõ code trên laptop phát sáng."""
        sit_py = gy - thl * 0.7
        pelvis = (gx, sit_py)
        torso_lean = 8.0 * s
        neck = (gx + torso_lean, pelvis[1] - tl * 0.9)
        head = (neck[0] + 6 * s, neck[1] - hr * 1.05)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Đùi nằm ngang, chân chạm đất
        l_foot = (l_hip[0] + 20 * s, gy)
        r_foot = (r_hip[0] + 24 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Bàn phím laptop đặt phía trước
        laptop_x = gx + 36 * s
        laptop_y = sit_py - 12 * s

        # Hai tay gõ bàn phím nhịp nhàng
        type_l = math.sin(t * 18.0) * (3.0 * s)
        type_r = math.cos(t * 18.0) * (3.0 * s)
        l_hand = (laptop_x - 4 * s, laptop_y + type_l)
        r_hand = (laptop_x + 8 * s, laptop_y + type_r)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            prop_type="laptop",
            prop_pos=(laptop_x, laptop_y),
            eye_offset=(4 * s, 3 * s),
        )

    def _pose_celebrate(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose ăn mừng chiến thắng: giơ nắm đấm lên trời."""
        bob = math.sin(t * 5.0) * (4.0 * s)
        pelvis = (gx, py + bob)
        neck = (gx, pelvis[1] - tl)
        head = (gx, neck[1] - hr * 1.1)

        l_sh = (neck[0] - sw, neck[1] + 4 * s)
        r_sh = (neck[0] + sw, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        l_foot = (l_hip[0] - 14 * s, gy)
        r_foot = (r_hip[0] + 14 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Tay phải giơ nắm đấm cao
        r_hand = (r_sh[0] + 16 * s, neck[1] - 36 * s)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        # Tay trái co gập ngang ngực thể hiện quyết tâm
        l_hand = (l_sh[0] + 6 * s, neck[1] + 12 * s)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            aura_intensity=0.8,
            aura_color="#39ff14",
            eye_offset=(2 * s, -4 * s),
        )

    def _pose_pushing(self, gx, py, gy, s, t, hr, tl, ual, fal, thl, shl, sw, hw, color):
        """Pose gồng mình đẩy vật cản / vật cản lớn (Struggle)."""
        effort = math.sin(t * 4.0) * (3.0 * s)
        pelvis = (gx - 18 * s + effort, py + 8 * s)
        neck = (gx + 12 * s + effort, pelvis[1] - tl * 0.75)
        head = (neck[0] + 12 * s, neck[1] - hr * 0.9)

        l_sh = (neck[0] - sw * 0.5, neck[1] + 4 * s)
        r_sh = (neck[0] + sw * 0.5, neck[1] + 4 * s)
        l_hip = (pelvis[0] - hw, pelvis[1])
        r_hip = (pelvis[0] + hw, pelvis[1])

        # Chân trước đạp mạnh, chân sau duỗi thẳng chịu lực
        l_foot = (gx - 32 * s, gy)
        r_foot = (gx + 10 * s, gy)
        (l_knee, _) = solve_ik_2bone(l_hip, l_foot, thl, shl, bend_dir=1)
        (r_knee, _) = solve_ik_2bone(r_hip, r_foot, thl, shl, bend_dir=1)

        # Hai tay đẩy thẳng về phía trước
        wall_x = gx + 42 * s
        l_hand = (wall_x, neck[1] + 4 * s)
        r_hand = (wall_x, neck[1] + 18 * s)
        (l_elb, _) = solve_ik_2bone(l_sh, l_hand, ual, fal, bend_dir=-1)
        (r_elb, _) = solve_ik_2bone(r_sh, r_hand, ual, fal, bend_dir=1)

        return StickmanPoseData(
            pelvis=pelvis, neck=neck, head=head, head_radius=hr,
            left_shoulder=l_sh, left_elbow=l_elb, left_hand=l_hand,
            right_shoulder=r_sh, right_elbow=r_elb, right_hand=r_hand,
            left_hip=l_hip, left_knee=l_knee, left_foot=l_foot,
            right_hip=r_hip, right_knee=r_knee, right_foot=r_foot,
            aura_intensity=0.6,
            aura_color="#ef4444",
            eye_offset=(4 * s, 1 * s),
        )


# Singleton mặc định
DEFAULT_STICKMAN_RIG = StickmanRig()
