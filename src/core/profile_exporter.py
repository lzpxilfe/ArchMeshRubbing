"""
2D 실루엣 추출 및 SVG 내보내기 모듈 (v5 - 디버깅 완료)
메쉬를 6방향에서 투영하여 외곽선을 폴리라인으로 추출합니다.
"""
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker polyline simplification.

    Args:
        points: (N, 2) polyline points
        epsilon: max perpendicular distance (same unit as points)
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]
    line = end - start
    line_len = float(np.hypot(line[0], line[1]))
    if line_len < 1e-12:
        dists = np.hypot(*(points - start).T)
    else:
        # 2D "cross product magnitude" gives twice triangle area => perpendicular distance.
        dists = np.abs(line[0] * (points[:, 1] - start[1]) - line[1] * (points[:, 0] - start[0])) / line_len

    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _rdp_simplify(points[: idx + 1], epsilon)
        right = _rdp_simplify(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _trace_main_contour_binary(mask: np.ndarray) -> np.ndarray:
    """
    Extract an ordered outer contour from a binary mask without OpenCV.

    Returns:
        (N, 2) float array in pixel coordinates (x, y), y-down.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for contour extraction when OpenCV is not available")

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float64)

    labels, num = ndimage.label(mask)
    if num <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros((0, 2), dtype=np.float64)
    counts[0] = 0
    main_label = int(np.argmax(counts))
    comp = labels == main_label

    # Find boundary pixels
    eroded = ndimage.binary_erosion(comp, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = comp & ~eroded
    coords = np.argwhere(boundary)
    if coords.size == 0:
        # Fallback: single pixel/component
        coords = np.argwhere(comp)
        if coords.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

    # Choose a deterministic start: top-most, then left-most.
    start_idx = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
    sy, sx = coords[start_idx]

    # Pad to simplify neighbor checks.
    comp_p = np.pad(comp, 1, mode='constant', constant_values=False)
    sy += 1
    sx += 1

    # Moore-neighbor tracing around the component (8-connectivity).
    # Directions in clockwise order starting from NW.
    dirs = np.array(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ],
        dtype=int,
    )
    dir_map = {(int(dy), int(dx)): i for i, (dy, dx) in enumerate(dirs)}

    b0 = (int(sy), int(sx))
    b = b0
    c0 = (b0[0], b0[1] - 1)  # west neighbor
    c = c0

    contour_xy: list[list[float]] = []
    max_steps = int(comp_p.size)  # generous upper bound

    for _ in range(max_steps):
        # store (x, y) in unpadded coordinates
        contour_xy.append([float(b[1] - 1), float(b[0] - 1)])

        dy = c[0] - b[0]
        dx = c[1] - b[1]
        d = dir_map.get((int(dy), int(dx)), 7)

        found = False
        # Scan neighbors clockwise, starting just after the backtrack direction.
        for step in range(1, 9):
            dn = (d + step) % 8
            nb = (b[0] + int(dirs[dn][0]), b[1] + int(dirs[dn][1]))
            if comp_p[nb]:
                # The "previous" neighbor is the one before nb in scanning order.
                prev_dir = (dn - 1) % 8
                c = (b[0] + int(dirs[prev_dir][0]), b[1] + int(dirs[prev_dir][1]))
                b = nb
                found = True
                break

        if not found:
            break

        if b == b0 and c == c0:
            break

    return np.asarray(contour_xy, dtype=np.float64)


def _extract_main_contour(occupancy: np.ndarray) -> np.ndarray:
    """
    occupancy: (H, W) uint8 image where 0=background, 255=filled.
    Returns (N, 2) float points in pixel coords (x, y), y-down.
    """
    if occupancy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if HAS_CV2:
        found_contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not found_contours:
            return np.zeros((0, 2), dtype=np.float64)
        main_cnt = max(found_contours, key=cv2.contourArea)
        epsilon = 0.0002 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        return approx.reshape(-1, 2).astype(np.float64)

    contour = _trace_main_contour_binary(occupancy > 0)
    if len(contour) < 3:
        return contour

    # Simplify (closed polygon)
    closed = np.vstack([contour, contour[0]])
    perimeter = float(np.sum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))))
    epsilon = max(0.5, 0.0002 * perimeter)
    simplified = _rdp_simplify(closed, epsilon)
    if len(simplified) >= 2 and np.allclose(simplified[0], simplified[-1]):
        simplified = simplified[:-1]
    return simplified


class ProfileExporter:
    """메쉬의 2D 실루엣을 추출하고 SVG로 내보냅니다."""
    
    # 6방향 뷰 정의
    # axes: (SVG X축에 매핑될 3D축 인덱스, SVG Y축에 매핑될 3D축 인덱스)
    # 모든 뷰에서 SVG 좌표계: 오른쪽=+X, 위쪽=+Y (표준)
    VIEWS = {
        'top':    {'axes': (0, 1)},  # XY 평면, X→오른쪽, Y→위쪽
        'bottom': {'axes': (0, 1)},  # XY 평면, X→오른쪽, Y→아래쪽
        'front':  {'axes': (0, 2)},  # XZ 평면, X→오른쪽, Z→위쪽
        'back':   {'axes': (0, 2)},  # XZ 평면, X→왼쪽, Z→위쪽
        'left':   {'axes': (1, 2)},  # YZ 평면, Y→왼쪽, Z→위쪽
        'right':  {'axes': (1, 2)},  # YZ 평면, Y→오른쪽, Z→위쪽
    }
    
    MAX_FACES_FOR_RASTERIZE = 100000  # 래스터화 최대 면 수
    
    def __init__(self, resolution: int = 1024):
        self.resolution = resolution
    
    def extract_silhouette(self, mesh, view: str = 'top', 
                           translation: np.ndarray = None,
                           rotation: np.ndarray = None,
                           scale: float = 1.0,
                           opengl_matrices: tuple = None,
                           viewport_image: Image.Image = None) -> tuple:
        """
        메쉬를 투영하여 단일 외곽선(폴리라인)을 추출합니다.
        opengl_matrices: (modelview, projection, viewport) - 제공 시 실제 렌더링 화면과 완벽히 일치시킴
        viewport_image: OpenGL 캡처 이미지(PIL). 제공 시 화면과 100% 일치하는 외곽선을 이미지 기반으로 추출
        """
        if opengl_matrices:
            mv, proj, vp = opengl_matrices
            mvp = mv @ proj

            # 1) 월드 bounds 계산 (전체 정점 변환 없이 bounds corner 8개만 변환)
            w_min = None
            w_max = None
            try:
                lb = np.asarray(mesh.bounds, dtype=np.float64)
                corners = np.array(
                    [
                        [lb[0, 0], lb[0, 1], lb[0, 2]],
                        [lb[1, 0], lb[0, 1], lb[0, 2]],
                        [lb[0, 0], lb[1, 1], lb[0, 2]],
                        [lb[1, 0], lb[1, 1], lb[0, 2]],
                        [lb[0, 0], lb[0, 1], lb[1, 2]],
                        [lb[1, 0], lb[0, 1], lb[1, 2]],
                        [lb[0, 0], lb[1, 1], lb[1, 2]],
                        [lb[1, 0], lb[1, 1], lb[1, 2]],
                    ],
                    dtype=np.float64,
                )
                corners = corners * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rotation, degrees=True)
                    corners = r.apply(corners)
                if translation is not None:
                    corners = corners + translation
                w_min = corners.min(axis=0)
                w_max = corners.max(axis=0)
            except Exception:
                pass

            if w_min is None or w_max is None:
                v_all = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rotation, degrees=True)
                    v_all = r.apply(v_all)
                if translation is not None:
                    v_all = v_all + translation
                w_min = v_all.min(axis=0)
                w_max = v_all.max(axis=0)

            # 2) px_per_cm 계산 (world 좌표 1.0을 1cm로 가정)
            def project_pt(p):
                vh = np.append(p, 1.0)
                vc = vh @ mvp
                if abs(float(vc[3])) < 1e-12:
                    return np.array([0.0, 0.0], dtype=np.float64)
                vn = vc[:3] / vc[3]
                return np.array([(vn[0] + 1) / 2 * vp[2], vp[3] - (vn[1] + 1) / 2 * vp[3]])

            center_world = (w_min + w_max) / 2.0
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if view in {"left", "right"}:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            p1 = center_world + axis

            px_per_cm = float(np.linalg.norm(project_pt(center_world) - project_pt(p1)))
            if px_per_cm < 1e-6:
                px_per_cm = 100.0  # Fallback

            # 3) 외곽선 추출: viewport_image가 있으면 이미지 기반(가장 견고/빠름)
            grid_w = int(vp[2])
            grid_h = int(vp[3])
            if viewport_image is not None:
                try:
                    img_rgba = viewport_image.convert("RGBA")
                    rgb = np.asarray(img_rgba, dtype=np.uint8)[..., :3]
                    if rgb.shape[1] != grid_w or rgb.shape[0] != grid_h:
                        # 캡처 사이즈가 vp와 다르면 vp를 이미지 기준으로 맞춤
                        grid_w = int(rgb.shape[1])
                        grid_h = int(rgb.shape[0])
                        vp = np.array([0, 0, grid_w, grid_h], dtype=np.int32)

                    corners_rgb = np.stack(
                        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0
                    ).astype(np.int16)
                    bg = np.median(corners_rgb, axis=0).astype(np.int16)

                    diff = np.max(np.abs(rgb.astype(np.int16) - bg[None, None, :]), axis=2)
                    mask = diff > 8
                    if int(mask.sum()) < 64:
                        # 매우 밝은 메쉬/조명 등으로 diff가 거의 없을 때: 단순 백색 임계값으로 fallback
                        mask = rgb.mean(axis=2) < 254

                    if HAS_CV2:
                        occ = (mask.astype(np.uint8) * 255)
                        k = np.ones((3, 3), dtype=np.uint8)
                        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k, iterations=2)
                        occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, k, iterations=1)
                        occupancy = occ
                    else:
                        if HAS_SCIPY:
                            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
                            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
                        occupancy = (mask.astype(np.uint8) * 255)
                except Exception:
                    viewport_image = None  # fallback to geometry rasterize

            if viewport_image is None:
                # (Fallback) geometry rasterize 기반 외곽선 추출
                faces = getattr(mesh, "faces", None)
                if faces is None:
                    faces = []
                faces = np.asarray(faces, dtype=np.int32)
                if faces.size == 0:
                    bounds = {
                        'min': np.array([0, 0]),
                        'max': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                        'size': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                        'px_per_cm': px_per_cm,
                        'is_pixels': True,
                        'vp_size': (grid_w, grid_h),
                        'grid_size': (grid_w, grid_h),
                        'matrices': (mv, proj, vp),
                        'world_bounds': np.array([w_min, w_max], dtype=np.float64),
                    }
                    return [], bounds

                if len(faces) > self.MAX_FACES_FOR_RASTERIZE:
                    step = max(1, int(len(faces) // self.MAX_FACES_FOR_RASTERIZE))
                    faces = faces[::step]
                    if len(faces) > self.MAX_FACES_FOR_RASTERIZE:
                        faces = faces[: self.MAX_FACES_FOR_RASTERIZE]

                unique_idx = np.unique(faces.reshape(-1))
                faces = np.searchsorted(unique_idx, faces).astype(np.int32, copy=False)

                vertices = np.asarray(mesh.vertices[unique_idx], dtype=np.float64) * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rotation, degrees=True)
                    vertices = r.apply(vertices)
                if translation is not None:
                    vertices = vertices + translation

                v_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
                v_clip = v_homo @ mvp  # (N, 4)
                v_ndc = v_clip[:, :3] / v_clip[:, 3:]
                proj_2d = np.zeros((len(vertices), 2))
                proj_2d[:, 0] = (v_ndc[:, 0] + 1) / 2 * grid_w
                proj_2d[:, 1] = (v_ndc[:, 1] + 1) / 2 * grid_h
                proj_2d[:, 1] = grid_h - proj_2d[:, 1]

                if HAS_CV2:
                    occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
                    img_v = proj_2d.astype(np.int32)
                    for face in faces:
                        cv2.fillPoly(occupancy, [img_v[face]], 255)
                else:
                    img = Image.new('L', (grid_w, grid_h), 0)
                    draw = ImageDraw.Draw(img)
                    for face in faces:
                        draw.polygon([(int(p[0]), int(p[1])) for p in proj_2d[face]], fill=255)
                    occupancy = np.array(img)

            contours = []
            contour_px = _extract_main_contour(occupancy)
            if len(contour_px) > 0:
                contours.append(contour_px.astype(float))

            bounds = {
                'min': np.array([0, 0]),
                'max': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                'size': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                'px_per_cm': px_per_cm,
                'is_pixels': True,
                'vp_size': (grid_w, grid_h),
                'grid_size': (grid_w, grid_h),
                'matrices': (mv, proj, vp),
                'world_bounds': np.array([w_min, w_max], dtype=np.float64),
            }
            return contours, bounds

        # Fallback (Orthogonal-like simple projection)
        if view not in self.VIEWS:
            raise ValueError(f"Unknown view: {view}")
        
        view_config = self.VIEWS[view]
        ax0, ax1 = view_config['axes']
        
        # 정점 변환
        vertices = mesh.vertices.copy() * scale
        if rotation is not None and np.any(rotation != 0):
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', rotation, degrees=True)
            vertices = r.apply(vertices)
        if translation is not None:
            vertices = vertices + translation
        
        # 2D 투영 (실제 좌표)
        proj_2d = vertices[:, [ax0, ax1]]
        if view == 'back': proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'left': proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'bottom': proj_2d[:, 1] = -proj_2d[:, 1]
        
        # 바운딩 박스 (실제 좌표)
        min_coords = proj_2d.min(axis=0)
        max_coords = proj_2d.max(axis=0)
        size = max_coords - min_coords
        margin = size * 0.02
        min_coords -= margin
        max_coords += margin
        size = max_coords - min_coords
        
        grid_res = max(self.resolution, 2048)
        aspect = size[0] / size[1] if size[1] > 0 else 1.0
        if aspect >= 1:
            grid_w, grid_h = grid_res, max(1, int(grid_res / aspect))
        else:
            grid_h, grid_w = grid_res, max(1, int(grid_res * aspect))
        
        def real_to_img(pts):
            img_pts = np.zeros_like(pts)
            img_pts[:, 0] = (pts[:, 0] - min_coords[0]) / size[0] * (grid_w - 1)
            img_pts[:, 1] = (1 - (pts[:, 1] - min_coords[1]) / size[1]) * (grid_h - 1)
            return img_pts

        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
        if HAS_CV2:
            img_v = real_to_img(proj_2d).astype(np.int32)
            for face in mesh.faces: cv2.fillPoly(occupancy, [img_v[face]], 255)
        else:
            img = Image.new('L', (grid_w, grid_h), 0)
            draw = ImageDraw.Draw(img)
            img_v = real_to_img(proj_2d)
            for face in mesh.faces: 
                draw.polygon([(int(p[0]), int(p[1])) for p in img_v[face]], fill=255)
            occupancy = np.array(img)

        contour_px = _extract_main_contour(occupancy)
        contours = []
        if len(contour_px) > 0:
            img_pts = contour_px.astype(float)

            # 이미지 좌표 -> 실제 좌표 변환
            real_pts = np.zeros_like(img_pts)
            real_pts[:, 0] = img_pts[:, 0] / (grid_w - 1) * size[0] + min_coords[0]
            real_pts[:, 1] = (1 - img_pts[:, 1] / (grid_h - 1)) * size[1] + min_coords[1]
            contours.append(real_pts)
        
        bounds = {
            'min': min_coords, 'max': max_coords, 'size': size,
            'grid_size': (grid_w, grid_h), 'is_pixels': False
        }
        return contours, bounds
    
    def generate_composite_image(self, viewport_image: Image.Image, bounds: dict,
                                 grid_spacing: float = 1.0) -> Image.Image:
        """
        사용자가 실제 본 화면(viewport_image) + 격자 오버레이 이미지 생성 (Multiply)
        """
        img_w, img_h = viewport_image.size
        grid_img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid_img)
        line_color = (200, 200, 200)

        if bounds.get('is_pixels') and 'matrices' in bounds:
            # 1. 행렬 기반 격자 투영
            mv, proj, vp = bounds['matrices']
            
            # 투류용 헬퍼
            def w_to_i(wx, wy, wz=0):
                vh = np.array([wx, wy, wz, 1.0])
                vc = vh @ (mv @ proj)
                if abs(vc[3]) < 1e-6: return None
                vn = vc[:3] / vc[3]
                return (vn[0]+1)/2 * img_w, img_h - (vn[1]+1)/2 * img_h

            # 가시 범위 계산 (메쉬 주변)
            if 'world_bounds' in bounds:
                wb = np.asarray(bounds['world_bounds'], dtype=np.float64)
                if wb.shape == (2, 3):
                    w_min, w_max = wb[0], wb[1]
                else:
                    w_min, w_max = wb.min(axis=0), wb.max(axis=0)
            else:
                v_world = bounds['vertices_world']
                w_min, w_max = v_world.min(axis=0), v_world.max(axis=0)
            pad = 20.0 # 20cm 여유
            
            # 세로선 (X축)
            x_start = np.floor((w_min[0] - pad) / grid_spacing) * grid_spacing
            x_end = np.ceil((w_max[0] + pad) / grid_spacing) * grid_spacing
            y_start, y_end = w_min[1] - pad, w_max[1] + pad
            
            x = x_start
            while x <= x_end:
                p1 = w_to_i(x, y_start); p2 = w_to_i(x, y_end)
                if p1 and p2: draw.line([p1, p2], fill=line_color, width=1)
                x += grid_spacing
                
            # 가로선 (Y축)
            y_start_grid = np.floor((w_min[1] - pad) / grid_spacing) * grid_spacing
            y_end_grid = np.ceil((w_max[1] + pad) / grid_spacing) * grid_spacing
            x_start_f, x_end_f = w_min[0] - pad, w_max[0] + pad
            
            y = y_start_grid
            while y <= y_end_grid:
                p1 = w_to_i(x_start_f, y); p2 = w_to_i(x_end_f, y)
                if p1 and p2: draw.line([p1, p2], fill=line_color, width=1)
                y += grid_spacing
        else:
            # 2. 고전적 평면 투영 (정사영 베이스)
            size = bounds['size']
            min_coords = bounds['min']
            
            # 세로선 (X)
            x = np.ceil(min_coords[0] / grid_spacing) * grid_spacing
            while x <= min_coords[0] + size[0]:
                px = int((x - min_coords[0]) / size[0] * (img_w - 1))
                if 0 <= px < img_w: draw.line([(px, 0), (px, img_h - 1)], fill=line_color, width=1)
                x += grid_spacing
            # 가로선 (Y)
            y = np.ceil(min_coords[1] / grid_spacing) * grid_spacing
            while y <= min_coords[1] + size[1]:
                py = int((1 - (y - min_coords[1]) / size[1]) * (img_h - 1))
                if 0 <= py < img_h: draw.line([(0, py), (img_w - 1, py)], fill=line_color, width=1)
                y += grid_spacing
            
        # 3. Multiply Blend
        mesh_arr = np.array(viewport_image.convert('RGB'), dtype=np.float32) / 255.0
        grid_arr = np.array(grid_img, dtype=np.float32) / 255.0
        result_arr = (mesh_arr * grid_arr * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr)
    
    def export_svg(self, contours: list, bounds: dict,
                    background_image: Image.Image = None,
                    stroke_color: str = "#000000", # 기본 검정색
                    stroke_width: float = 0.05,    # 0.5mm
                    extra_paths: list = None,
                    output_path: str = None) -> str:
        """
        최종 SVG 생성 (배경 이미지 + 단일 벡터 외곽선)
        """
        size = bounds['size']
        min_coords = bounds['min']
        
        svg_width = size[0]
        svg_height = size[1]
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            f'<svg xmlns="http://www.w3.org/2000/svg" ',
            f'xmlns:xlink="http://www.w3.org/1999/xlink" ',
            f'width="{svg_width:.2f}cm" height="{svg_height:.2f}cm" ',
            f'viewBox="0 0 {svg_width:.6f} {svg_height:.6f}">',
            f'<!-- Produced by ArchMeshRubbing - Resolution: {background_image.size if background_image else "Vector Only"} -->',
        ]
        
        # 1. 배경 이미지 삽입
        if background_image is not None:
            buffer = io.BytesIO()
            background_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            svg_parts.append(
                f'<image x="0" y="0" width="{svg_width:.6f}" height="{svg_height:.6f}" '
                f'preserveAspectRatio="none" '
                f'xlink:href="data:image/png;base64,{img_base64}" />'
            )
            
        # 2. 외곽선 벡터 삽입
        if contours:
            svg_parts.append(
                f'<g id="outline" stroke="{stroke_color}" fill="none" stroke-width="{stroke_width}" '
                f'stroke-linejoin="round" stroke-linecap="round">'
            )
            for contour in contours:
                if len(contour) < 2: continue
                # 좌표 변환
                svg_pts = contour.copy()
                
                # is_pixels (OpenGL 투영)인 경우 이미 Viewport 상단(0) 기준이므로 그대로 사용 가능하나
                # extract_silhouette에서 vp[3] - proj_2d[:, 1]로 이미 뒤집었는지 확인 필요
                # 현재 코드: proj_2d[:, 1] = vp[3] - proj_2d[:, 1] (라인 69) -> 이미지 좌표계(Top-Left 0)
                # 따라서 min_coords=0이므로 그대로 두면 됨.
                if bounds.get('is_pixels'):
                   # 픽셀 좌표 -> cm 단위로 스케일링만 수행
                   px_per_cm = bounds['px_per_cm']
                   svg_pts = svg_pts / px_per_cm
                else:
                   # 월드 좌표 (Top-Down 투영 등) -> Y축 반전 필요 (SVG는 Y가 아래로 증가, 월드는 위로 증가)
                   svg_pts[:, 0] -= min_coords[0]
                   svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])
                
                # CAD import 시 "한 개의 선(연속 폴리라인)"으로 들어오도록 단일 path로 출력
                d_parts = [f'M {svg_pts[0,0]:.6f} {svg_pts[0,1]:.6f}']
                for p in svg_pts[1:]:
                    d_parts.append(f'L {p[0]:.6f} {p[1]:.6f}')
                d_parts.append('Z')
                svg_parts.append(f'<path d="{" ".join(d_parts)}" />')
            svg_parts.append('</g>')

        # 3. 추가 선(단면 가이드 등) 삽입
        if extra_paths:
            svg_parts.append(
                '<g id="cut_lines" fill="none" stroke-linejoin="round" stroke-linecap="round">'
            )
            for item in extra_paths:
                try:
                    pts = np.asarray(item.get("points", []), dtype=np.float64)
                except Exception:
                    continue
                if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
                    continue

                stroke = str(item.get("stroke", "#ff4040"))
                sw = float(item.get("stroke_width", stroke_width))
                path_id = item.get("id", None)

                svg_pts = pts[:, :2].copy()
                if bounds.get('is_pixels'):
                    px_per_cm = float(bounds.get('px_per_cm', 100.0))
                    if px_per_cm <= 0:
                        px_per_cm = 100.0
                    svg_pts = svg_pts / px_per_cm
                else:
                    svg_pts[:, 0] -= min_coords[0]
                    svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])

                d_parts = [f'M {svg_pts[0,0]:.6f} {svg_pts[0,1]:.6f}']
                for p in svg_pts[1:]:
                    d_parts.append(f'L {p[0]:.6f} {p[1]:.6f}')

                id_attr = f' id="{path_id}"' if path_id else ""
                svg_parts.append(
                    f'<path{id_attr} d="{" ".join(d_parts)}" stroke="{stroke}" stroke-width="{sw}" />'
                )
            svg_parts.append('</g>')
             
        svg_parts.append('</svg>')
        
        content = '\n'.join(svg_parts)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        return content
        
    def export_profile(self, mesh, view: str, output_path: str,
                       translation: np.ndarray = None,
                       rotation: np.ndarray = None,
                       scale: float = 1.0,
                       grid_spacing: float = 1.0,
                       include_grid: bool = True,
                       viewport_image: Image.Image = None,
                       opengl_matrices: tuple = None,
                       cut_lines_world: list = None) -> str:
        """
        메쉬의 2D 프로파일을 SVG로 내보내는 통합 메서드.
        """
        contours, bounds = self.extract_silhouette(
            mesh,
            view,
            translation,
            rotation,
            scale,
            opengl_matrices,
            viewport_image=viewport_image,
        )
        
        background_image = None
        if include_grid:
            if viewport_image:
                # 캡처된 실제 이미지와 격자 합성
                background_image = self.generate_composite_image(
                    viewport_image, bounds, grid_spacing
                )
            else:
                # (Fallback) 실루엣과 격자 합성 - occupancy가 없으므로 빈 이미지 배경
                dummy_viewport = Image.new('RGB', bounds['grid_size'], (255, 255, 255))
                background_image = self.generate_composite_image(
                    dummy_viewport, bounds, grid_spacing
                )

        extra_paths = []
        if cut_lines_world and opengl_matrices and view in {"top", "bottom"}:
            try:
                mv, proj, vp = opengl_matrices
                mvp = mv @ proj

                def project_world_to_px(pts_world: np.ndarray) -> np.ndarray:
                    pts_world = np.asarray(pts_world, dtype=np.float64)
                    if pts_world.ndim != 2:
                        return np.zeros((0, 2), dtype=np.float64)
                    if pts_world.shape[1] == 2:
                        pts_world = np.hstack([pts_world, np.zeros((len(pts_world), 1), dtype=np.float64)])
                    v_homo = np.hstack([pts_world[:, :3], np.ones((len(pts_world), 1), dtype=np.float64)])
                    v_clip = v_homo @ mvp
                    v_ndc = v_clip[:, :3] / v_clip[:, 3:]
                    x = (v_ndc[:, 0] + 1.0) / 2.0 * float(vp[2])
                    y = float(vp[3]) - (v_ndc[:, 1] + 1.0) / 2.0 * float(vp[3])
                    return np.stack([x, y], axis=1)

                colors = ["#ff4040", "#2b8cff"]
                for i, line in enumerate(cut_lines_world):
                    if not line or len(line) < 2:
                        continue
                    pts_w = np.asarray(line, dtype=np.float64)
                    pts_px = project_world_to_px(pts_w)
                    if pts_px.shape[0] < 2:
                        continue
                    extra_paths.append(
                        {
                            "id": f"cut_line_{i+1}",
                            "points": pts_px,
                            "stroke": colors[i % len(colors)],
                            "stroke_width": 0.015,  # 0.15mm
                        }
                    )
            except Exception:
                extra_paths = []

        return self.export_svg(
            contours, bounds, background_image,
            stroke_width=0.015,  # 0.15mm
            extra_paths=extra_paths,
            output_path=output_path
        )
