"""
Surface Visualizer Module
표면 시각화 - 텍스처 없는 메쉬용 탁본 효과 생성

깊이맵, 노멀맵, 곡률맵 등을 이용해 탁본과 유사한 이미지를 생성합니다.
"""

from dataclasses import dataclass
import numpy as np
from PIL import Image
from scipy import ndimage

from .flattener import FlattenedMesh


@dataclass
class RubbingImage:
    """
    탁본 이미지 결과
    
    Attributes:
        image: 탁본 이미지 (numpy array)
        width_real: 실제 너비
        height_real: 실제 높이
        unit: 단위
        dpi: 해상도
    """
    image: np.ndarray
    width_real: float
    height_real: float
    unit: str = 'mm'
    dpi: int = 300
    
    @property
    def width_pixels(self) -> int:
        return self.image.shape[1]
    
    @property
    def height_pixels(self) -> int:
        return self.image.shape[0]
    
    @property
    def pixels_per_unit(self) -> float:
        """단위당 픽셀 수"""
        try:
            w = float(self.width_real)
        except Exception:
            return 0.0
        if not np.isfinite(w) or w <= 1e-12:
            return 0.0
        return float(self.width_pixels) / w
    
    def to_pil_image(self) -> Image.Image:
        """PIL Image로 변환"""
        if self.image.dtype != np.uint8:
            img = self.image.astype(np.float64)
            img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
            img = img.astype(np.uint8)
        else:
            img = self.image
        
        if len(img.shape) == 2:
            return Image.fromarray(img, mode='L')
        else:
            return Image.fromarray(img)
    
    def save(self, filepath: str, include_scale_bar: bool = True) -> None:
        """
        이미지 저장
        
        Args:
            filepath: 저장 경로
            include_scale_bar: 스케일 바 포함 여부
        """
        img = self.to_pil_image()
        
        if include_scale_bar:
            img = self._add_scale_bar(img)
        
        img.save(filepath, dpi=(self.dpi, self.dpi))
    
    def _add_scale_bar(self, img: Image.Image) -> Image.Image:
        """스케일 바 추가"""
        from PIL import ImageDraw, ImageFont

        ppu = float(self.pixels_per_unit)
        if not np.isfinite(ppu) or ppu <= 1e-12:
            return img
        
        # 이미지 복사
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 스케일 바 크기 계산 (이미지 너비의 10-20%)
        target_bar_ratio = 0.15
        target_bar_pixels = int(img.width * target_bar_ratio)
        
        # 실제 크기에 맞는 깔끔한 숫자로 조정
        bar_real_size = float(target_bar_pixels) / ppu
        
        # 깔끔한 숫자로 반올림 (1, 2, 5, 10, 20, 50, 100...)
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        magnitude = 10 ** np.floor(np.log10(bar_real_size))
        normalized = bar_real_size / magnitude
        
        nice_normalized = min(nice_values, key=lambda x: abs(x - normalized))
        nice_bar_size = nice_normalized * magnitude
        
        # 실제 픽셀 크기
        bar_pixels = int(float(nice_bar_size) * ppu)
        
        # 스케일 바 위치 (오른쪽 하단)
        margin = 20
        bar_height = 10
        bar_x = img.width - margin - bar_pixels
        bar_y = img.height - margin - bar_height - 20
        
        # 배경
        draw.rectangle([bar_x - 5, bar_y - 5, 
                       bar_x + bar_pixels + 5, bar_y + bar_height + 25],
                      fill='white', outline='black')
        
        # 스케일 바
        draw.rectangle([bar_x, bar_y, bar_x + bar_pixels, bar_y + bar_height],
                      fill='black')
        
        # 텍스트
        label = f"{nice_bar_size:.0f} {self.unit}"
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bar_x + (bar_pixels - text_width) // 2
        draw.text((text_x, bar_y + bar_height + 3), label, fill='black', font=font)
        
        return img


class SurfaceVisualizer:
    """
    표면 시각화 도구
    
    텍스처가 없는 메쉬에서 탁본과 유사한 이미지를 생성합니다.
    """
    
    def __init__(self, default_dpi: int = 300):
        """
        Args:
            default_dpi: 기본 출력 해상도
        """
        self.default_dpi = default_dpi
    
    def generate_rubbing(
        self,
        flattened: FlattenedMesh,
        width_pixels: int = 2000,
        style: str = 'traditional',
        light_angle: float = 45.0,
        *,
        height_mode: str = "normal_z",
        remove_curvature: bool = False,
        reference_sigma: float | None = None,
        relief_strength: float = 1.0,
        invert: bool = False,
        preset: str | None = None,
        image_mode: str = "mesh",
        smooth_sigma: float = 0.0,
        detail_strength: float = 1.0,
        detail_sigma: float | None = None,
    ) -> RubbingImage:
        """
        탁본 효과 이미지 생성
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비 (픽셀)
            style: 스타일 ('traditional', 'modern', 'relief')
            light_angle: 조명 각도 (도)
            height_mode: 높이값 소스 ('normal_z'|'axis').
            remove_curvature: 곡률(저주파)을 제거하여 디지털 탁본처럼 만듭니다.
            reference_sigma: 곡률 제거용 가우시안 sigma(px). None이면 해상도에 맞춰 자동.
            relief_strength: 요철(잔무늬) 강조 배율.
            invert: 높이 부호 반전(내면/외면 톤 뒤집기 등에 사용).
             
            preset: 스타일 프리셋 이름.
            image_mode: 'mesh' | 'image' (이미지 기반 펼침).
            smooth_sigma: 이미지 기반 렌더링 시 부드럽게 하는 정도(px).
            detail_strength: 디테일 강조/완화 정도 (1.0=기본).
            detail_sigma: 디테일 추출용 블러 sigma(px). None이면 자동.

        Returns:
            RubbingImage: 탁본 이미지
        """
        width_pixels = int(width_pixels)
        if width_pixels < 1:
            width_pixels = 1

        preset_cfg = self._resolve_rubbing_preset(preset)
        if preset_cfg:
            style = preset_cfg.get("style", style)
            image_mode = preset_cfg.get("image_mode", image_mode)
            smooth_sigma = preset_cfg.get("smooth_sigma", smooth_sigma)
            detail_strength = preset_cfg.get("detail_strength", detail_strength)
            detail_sigma = preset_cfg.get("detail_sigma", detail_sigma)
            remove_curvature = preset_cfg.get("remove_curvature", remove_curvature)
            reference_sigma = preset_cfg.get("reference_sigma", reference_sigma)
            relief_strength = preset_cfg.get("relief_strength", relief_strength)
            height_mode = preset_cfg.get("height_mode", height_mode)
            invert = preset_cfg.get("invert", invert)
            splat_sigma = preset_cfg.get("splat_sigma", None)
        else:
            splat_sigma = None

        if flattened is None or getattr(flattened, "uv", None) is None or getattr(flattened, "faces", None) is None:
            raise ValueError("Flattened mesh is missing uv/faces.")
        if flattened.uv.ndim != 2 or flattened.uv.shape[0] == 0:
            raise ValueError("Flattened mesh has no UV vertices.")
        if flattened.faces.ndim != 2 or flattened.faces.shape[0] == 0:
            raise ValueError("Flattened mesh has no faces.")

        # 이미지 크기 계산
        w_real = float(flattened.width)
        h_real = float(flattened.height)
        if not np.isfinite(w_real) or not np.isfinite(h_real) or w_real <= 1e-12:
            raise ValueError("Flattened mesh has invalid dimensions.")
        aspect_ratio = h_real / w_real
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        height_pixels = max(1, int(width_pixels * aspect_ratio))

        # Safety: cap total pixels to avoid OOM on huge unwraps.
        try:
            max_total = int(getattr(self, "_max_total_pixels", 20000000) or 20000000)
        except Exception:
            max_total = 20000000
        max_total = max(1000000, int(max_total))
        total = int(width_pixels) * int(height_pixels)
        if total > max_total:
            scale = float(np.sqrt(float(max_total) / float(max(1, total))))
            width_pixels = max(1, int(float(width_pixels) * scale))
            height_pixels = max(1, int(float(height_pixels) * scale))
        
        # Value/height map (image-based pipeline)
        values = self._compute_height_values(flattened, mode=str(height_mode))
        mode = str(image_mode or "mesh").strip().lower()
        if mode in {"image", "baked", "bake", "img"}:
            depth_map = self._create_value_map_image_based(
                flattened,
                values,
                width_pixels,
                height_pixels,
                smooth_sigma=smooth_sigma,
                splat_sigma=splat_sigma,
            )
        else:
            depth_map = self._create_value_map(flattened, values, width_pixels, height_pixels)

        if bool(invert):
            depth_map = -depth_map

        if bool(remove_curvature):
            sigma_val = reference_sigma
            if sigma_val is None:
                sigma_val = max(2.0, 0.02 * float(min(width_pixels, height_pixels)))
            try:
                sigma_f = float(sigma_val)
            except Exception:
                sigma_f = 0.0
            if np.isfinite(sigma_f) and sigma_f > 0.0:
                try:
                    ref = ndimage.gaussian_filter(depth_map, sigma=float(sigma_f))
                    depth_map = depth_map - ref
                except Exception:
                    pass

        depth_map = self._apply_detail_enhancement(
            depth_map,
            strength=detail_strength,
            sigma=detail_sigma,
        )

        try:
            strength = float(relief_strength)
        except Exception:
            strength = 1.0
        if np.isfinite(strength) and abs(strength - 1.0) > 1e-12:
            depth_map = depth_map * strength
        
        # 스타일에 따른 렌더링
        if style == 'traditional':
            image = self._render_traditional_rubbing(depth_map, light_angle)
        elif style == 'modern':
            image = self._render_modern_rubbing(depth_map, light_angle)
        elif style == 'relief':
            image = self._render_relief(depth_map, light_angle)
        else:
            image = self._render_traditional_rubbing(depth_map, light_angle)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )

    def _resolve_rubbing_preset(self, preset: str | None) -> dict | None:
        if preset is None:
            return None
        key = str(preset).strip()
        if not key:
            return None

        presets = {
            "자연(이미지)": dict(
                style="traditional",
                image_mode="image",
                smooth_sigma=1.2,
                detail_strength=1.2,
                detail_sigma=None,
                splat_sigma=0.4,
            ),
            "선명(이미지)": dict(
                style="modern",
                image_mode="image",
                smooth_sigma=0.6,
                detail_strength=1.8,
                detail_sigma=None,
                splat_sigma=0.3,
            ),
            "부드러움": dict(
                style="traditional",
                image_mode="image",
                smooth_sigma=2.0,
                detail_strength=0.8,
                detail_sigma=None,
                splat_sigma=0.6,
            ),
            "디지털(곡률 제거)": dict(
                style="modern",
                image_mode="image",
                smooth_sigma=1.0,
                detail_strength=1.4,
                detail_sigma=None,
                splat_sigma=0.4,
                remove_curvature=True,
                reference_sigma=None,
                height_mode="cyl_radial",
                relief_strength=6.0,
            ),
            "레거시(메쉬)": dict(
                style="traditional",
                image_mode="mesh",
                smooth_sigma=0.0,
                detail_strength=1.0,
                detail_sigma=None,
            ),
        }

        if key in presets:
            return dict(presets[key])

        key_low = key.lower()
        for name, cfg in presets.items():
            if key_low in name.lower():
                return dict(cfg)
        return None

    def _estimate_thickness_axis(self, vertices: np.ndarray) -> np.ndarray:
        """PCA 기반 두께(시트 법선) 축 추정. (얇은 쉘/기와에 안정적)"""
        v = np.asarray(vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] < 8 or v.shape[1] < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = v[:, :3]
        finite = np.isfinite(v).all(axis=1)
        v = v[finite]
        if v.shape[0] < 8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        c = np.mean(v, axis=0)
        x = v - c
        cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
        try:
            w, vecs = np.linalg.eigh(cov)
            axis = vecs[:, int(np.argsort(w)[0])]
        except Exception:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        n = float(np.linalg.norm(axis))
        if not np.isfinite(n) or n < 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = (axis / n).astype(np.float64, copy=False)

        # Prefer +Z to avoid random flips (good default after positioning).
        if float(axis[2]) < 0.0:
            axis = -axis
        return axis

    def _estimate_long_axis(self, vertices: np.ndarray) -> np.ndarray:
        """PCA 기반 길이(주축) 방향 추정. (원통/기와의 축에 가까운 방향)"""
        v = np.asarray(vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] < 8 or v.shape[1] < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = v[:, :3]
        finite = np.isfinite(v).all(axis=1)
        v = v[finite]
        if v.shape[0] < 8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        c = np.mean(v, axis=0)
        x = v - c
        cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
        try:
            w, vecs = np.linalg.eigh(cov)
            axis = vecs[:, int(np.argsort(w)[-1])]
        except Exception:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        n = float(np.linalg.norm(axis))
        if not np.isfinite(n) or n < 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = (axis / n).astype(np.float64, copy=False)

        if float(axis[2]) < 0.0:
            axis = -axis
        return axis

    def _compute_height_values(self, flattened: FlattenedMesh, *, mode: str) -> np.ndarray:
        """Compute per-vertex scalar values used for rubbing depth (height map)."""
        original = flattened.original_mesh
        vertices = np.asarray(getattr(original, "vertices", np.zeros((0, 3))), dtype=np.float64)
        n = int(vertices.shape[0]) if vertices.ndim == 2 else 0
        if n <= 0:
            return np.zeros((0,), dtype=np.float64)

        m = str(mode or "").strip().lower()
        if m in {"radial", "cyl_radial", "cylinder_radial", "radius"}:
            meta = getattr(flattened, "meta", None)
            axis = None
            center = None
            radius = None
            if isinstance(meta, dict):
                axis = meta.get("cylinder_axis", None)
                center = meta.get("cylinder_center", None)
                radius = meta.get("cylinder_radius", None)

            try:
                a = np.asarray(axis, dtype=np.float64).reshape(3) if axis is not None else None
            except Exception:
                a = None
            if a is None or (not np.isfinite(a).all()) or float(np.linalg.norm(a)) < 1e-12:
                a = self._estimate_long_axis(vertices)
            else:
                a = a / float(np.linalg.norm(a))

            t = vertices[:, :3] @ a.reshape(3,)
            perp = vertices[:, :3] - t[:, None] * a.reshape(1, 3)
            try:
                c = np.asarray(center, dtype=np.float64).reshape(3) if center is not None else None
            except Exception:
                c = None
            if c is None or (not np.isfinite(c).all()):
                c = perp.mean(axis=0)
            r_vec = perp - c.reshape(1, 3)
            r = np.linalg.norm(r_vec, axis=1)

            try:
                r0 = float(radius) if radius is not None else float("nan")
            except Exception:
                r0 = float("nan")
            if not np.isfinite(r0) or abs(r0) < 1e-12:
                rf = r[np.isfinite(r)]
                r0 = float(np.median(rf)) if rf.size else 1.0
            out = (r - float(r0)).astype(np.float64, copy=False)
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        if m in {"axis", "thickness", "pca"}:
            axis = self._estimate_thickness_axis(vertices)
            out = vertices[:, :3] @ axis.reshape(3,)
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out.astype(np.float64, copy=False)

        # Legacy: normals.z
        try:
            original.compute_normals()
        except Exception:
            pass
        normals = getattr(original, "normals", None)
        if normals is None:
            return np.zeros((n,), dtype=np.float64)
        nn = np.asarray(normals, dtype=np.float64)
        if nn.ndim != 2 or nn.shape[0] != n or nn.shape[1] < 3:
            return np.zeros((n,), dtype=np.float64)
        out = nn[:, 2].astype(np.float64, copy=False)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _create_value_map(
        self,
        flattened: FlattenedMesh,
        values: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Create an image value map (height/depth/etc.) from per-vertex scalar values.

        - Small meshes: triangle rasterization (accurate).
        - Large meshes: vertex splat (fast) + hole filling + light smoothing.
        """
        normalized = flattened.normalize()
        uv = np.asarray(normalized.uv, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if uv.ndim != 2 or uv.shape[0] == 0 or uv.shape[1] < 2:
            return np.zeros((int(height), int(width)), dtype=np.float32)
        if vals.size != int(uv.shape[0]):
            vals = np.zeros((int(uv.shape[0]),), dtype=np.float64)

        try:
            n_faces = int(getattr(flattened, "n_faces", 0) or 0)
        except Exception:
            n_faces = 0
        if n_faces <= 0:
            try:
                n_faces = int(getattr(flattened.faces, "shape", [0])[0])
            except Exception:
                n_faces = 0
        try:
            fast_thr = int(getattr(self, "_depthmap_fast_threshold_faces", 250000) or 250000)
        except Exception:
            fast_thr = 250000
        if n_faces >= max(10000, fast_thr):
            return self._create_depth_map_fast_vertex_splat(uv, vals, int(width), int(height))

        faces = np.asarray(flattened.faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
            return self._create_depth_map_fast_vertex_splat(uv, vals, int(width), int(height))

        # Depth buffer init
        w = int(max(1, width))
        h = int(max(1, height))
        buffer = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.int32)

        for face in faces:
            self._rasterize_triangle_with_values(
                uv[face], vals[face],
                w, h,
                buffer, count,
            )

        mask = count > 0
        buffer[mask] /= count[mask].astype(np.float32, copy=False)
        if not bool(mask.all()):
            buffer = self._fill_holes(buffer, mask)
        return buffer

    def _create_value_map_image_based(
        self,
        flattened: FlattenedMesh,
        values: np.ndarray,
        width: int,
        height: int,
        *,
        smooth_sigma: float = 0.0,
        splat_sigma: float | None = None,
    ) -> np.ndarray:
        """Image-based value map (vertex splat + optional smoothing)."""
        normalized = flattened.normalize()
        uv = np.asarray(normalized.uv, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if uv.ndim != 2 or uv.shape[0] == 0 or uv.shape[1] < 2:
            return np.zeros((int(height), int(width)), dtype=np.float32)
        if vals.size != int(uv.shape[0]):
            vals = np.zeros((int(uv.shape[0]),), dtype=np.float64)

        depth = self._create_depth_map_fast_vertex_splat(
            uv,
            vals,
            int(width),
            int(height),
            splat_sigma=splat_sigma,
        )

        try:
            sigma = float(smooth_sigma)
        except Exception:
            sigma = 0.0
        if np.isfinite(sigma) and sigma > 0.0:
            try:
                depth = ndimage.gaussian_filter(depth, sigma=float(sigma))
            except Exception:
                pass

        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth
    
    def _create_depth_map(self, flattened: FlattenedMesh,
                          width: int, height: int) -> np.ndarray:
        """
        평면화된 메쉬에서 깊이맵 생성
        
        로컬 높이 변화를 깊이로 사용합니다.
        """
        values = self._compute_height_values(flattened, mode="normal_z")
        return self._create_value_map(flattened, values, int(width), int(height))

    def _create_depth_map_fast_vertex_splat(
        self,
        uv: np.ndarray,
        heights: np.ndarray,
        width: int,
        height: int,
        *,
        splat_sigma: float | None = None,
    ) -> np.ndarray:
        """Fast depth-map creation by splatting per-vertex values into pixels.

        Notes
        -----
        For large meshes we avoid per-triangle rasterization. Instead we splat vertex
        values into the image using bilinear weights and apply a *normalized* blur:

            depth = G(sums) / (G(weights) + eps)

        This reduces the "pixel-grid" noise that happens with nearest-neighbor binning
        and fills small holes more naturally than ad-hoc dilation.
        """
        uv = np.asarray(uv, dtype=np.float64)
        h = np.asarray(heights, dtype=np.float64).reshape(-1)
        if uv.ndim != 2 or uv.shape[0] == 0 or uv.shape[1] < 2:
            return np.zeros((int(height), int(width)), dtype=np.float64)
        if h.size != int(uv.shape[0]):
            h = np.zeros((int(uv.shape[0]),), dtype=np.float64)

        w = int(max(1, width))
        hh = int(max(1, height))

        u = uv[:, 0].reshape(-1)
        v = uv[:, 1].reshape(-1)
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(h)
        if not bool(np.any(valid)):
            return np.zeros((hh, w), dtype=np.float32)

        u = u[valid]
        v = v[valid]
        hv = h[valid]

        # Map to pixel coords (UV is already normalized to [0,1] by FlattenedMesh.normalize()).
        fx = (u * float(w - 1)).astype(np.float64, copy=False)
        fy = ((1.0 - v) * float(hh - 1)).astype(np.float64, copy=False)

        x0 = np.floor(fx).astype(np.int64, copy=False)
        y0 = np.floor(fy).astype(np.int64, copy=False)
        dx = (fx - x0.astype(np.float64, copy=False)).astype(np.float64, copy=False)
        dy = (fy - y0.astype(np.float64, copy=False)).astype(np.float64, copy=False)

        x1 = x0 + 1
        y1 = y0 + 1
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, hh - 1)
        y1 = np.clip(y1, 0, hh - 1)

        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        lin00 = y0 * int(w) + x0
        lin10 = y0 * int(w) + x1
        lin01 = y1 * int(w) + x0
        lin11 = y1 * int(w) + x1

        lin_all = np.concatenate([lin00, lin10, lin01, lin11], axis=0)
        w_all = np.concatenate([w00, w10, w01, w11], axis=0)
        hv_all = np.concatenate([hv * w00, hv * w10, hv * w01, hv * w11], axis=0)

        size = int(w * hh)
        sums = np.bincount(lin_all, weights=hv_all, minlength=size).astype(np.float64, copy=False)
        weights = np.bincount(lin_all, weights=w_all, minlength=size).astype(np.float64, copy=False)

        sums_img = sums.reshape((hh, w))
        weights_img = weights.reshape((hh, w))

        # Normalized smoothing to reduce splat noise and fill small holes naturally.
        try:
            if splat_sigma is not None:
                sigma = float(splat_sigma)
            else:
                sigma = float(getattr(self, "_depthmap_fast_sigma", 1.0) or 0.0)
        except Exception:
            sigma = 1.0

        # Adapt smoothing when the requested resolution is very high relative to vertex density.
        try:
            density = float(hv.size) / float(max(1, w * hh))
        except Exception:
            density = 1.0
        if np.isfinite(density) and density > 0.0 and splat_sigma is None:
            if density < 0.08:
                sigma = max(float(sigma), 2.0)
            elif density < 0.20:
                sigma = max(float(sigma), 1.4)

        if np.isfinite(sigma) and sigma > 0.0:
            try:
                sums_img = ndimage.gaussian_filter(sums_img, sigma=float(sigma))
                weights_img = ndimage.gaussian_filter(weights_img, sigma=float(sigma))
            except Exception:
                pass

        eps = 1e-12
        depth = sums_img / (weights_img + float(eps))
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(depth, dtype=np.float32)

    def _apply_detail_enhancement(
        self,
        depth_map: np.ndarray,
        *,
        strength: float = 1.0,
        sigma: float | None = None,
    ) -> np.ndarray:
        try:
            s = float(strength)
        except Exception:
            s = 1.0
        if not np.isfinite(s) or abs(s - 1.0) < 1e-9:
            return depth_map

        h, w = depth_map.shape[:2]
        if sigma is None:
            sigma_val = max(1.0, 0.01 * float(min(w, h)))
        else:
            try:
                sigma_val = float(sigma)
            except Exception:
                sigma_val = 0.0
        if not np.isfinite(sigma_val) or sigma_val <= 0.0:
            return depth_map

        try:
            blur = ndimage.gaussian_filter(depth_map, sigma=float(sigma_val))
            detail = depth_map - blur
            out = depth_map + s * detail
        except Exception:
            return depth_map

        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
    
    def _rasterize_triangle_with_values(self, uv: np.ndarray, values: np.ndarray,
                                        width: int, height: int,
                                        buffer: np.ndarray, 
                                        count: np.ndarray) -> None:
        """값을 보간하며 삼각형 래스터화"""
        # 픽셀 좌표로 변환
        px = (uv[:, 0] * (width - 1)).astype(np.float64)
        py = ((1 - uv[:, 1]) * (height - 1)).astype(np.float64)  # Y 뒤집기
        
        # 바운딩 박스
        min_x = max(0, int(np.floor(px.min())))
        max_x = min(width - 1, int(np.ceil(px.max())))
        min_y = max(0, int(np.floor(py.min())))
        max_y = min(height - 1, int(np.ceil(py.max())))
        
        # 면적 계산
        v0 = np.array([px[0], py[0]])
        v1 = np.array([px[1], py[1]])
        v2 = np.array([px[2], py[2]])
        
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        if abs(area) < 1e-10:
            return
        
        # 삼각형 내부 픽셀 채우기
        for iy in range(min_y, max_y + 1):
            for ix in range(min_x, max_x + 1):
                p = np.array([ix + 0.5, iy + 0.5])
                
                # 무게중심 좌표
                w0 = ((v1[0] - p[0]) * (v2[1] - p[1]) - (v2[0] - p[0]) * (v1[1] - p[1])) / area
                w1 = ((v2[0] - p[0]) * (v0[1] - p[1]) - (v0[0] - p[0]) * (v2[1] - p[1])) / area
                w2 = 1 - w0 - w1
                
                # 삼각형 내부 체크
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # 값 보간
                    value = w0 * values[0] + w1 * values[1] + w2 * values[2]
                    buffer[iy, ix] += value
                    count[iy, ix] += 1
    
    def _fill_holes(self, image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """빈 영역을 주변 값으로 채움"""
        from scipy.ndimage import binary_dilation

        result = image.copy()
        
        # 유효한 영역 확장
        for _ in range(10):
            # 현재 마스크 확장
            dilated = binary_dilation(valid_mask)
            
            # 새로 추가된 영역
            new_valid = dilated & ~valid_mask
            
            if not new_valid.any():
                break
            
            # 주변 값으로 채우기
            for axis in range(2):
                # 각 방향으로 이동
                for direction in [-1, 1]:
                    shifted = np.roll(result, direction, axis=axis)
                    shifted_valid = np.roll(valid_mask, direction, axis=axis)
                    
                    # 새 영역에 값 할당
                    update_mask = new_valid & shifted_valid
                    result[update_mask] = shifted[update_mask]
            
            valid_mask = dilated
        
        return result
    
    def _render_traditional_rubbing(self, depth_map: np.ndarray,
                                     light_angle: float) -> np.ndarray:
        """
        전통적인 탁본 스타일 렌더링
        
        검은 바탕에 흰색/회색 음영
        """
        # 그라디언트 계산 (기울기)
        # NOTE: np.gradient는 depth-map의 미세 노이즈를 그대로 증폭시키는 경향이 있어,
        #       Sobel(=가벼운 스무딩+미분)로 더 안정적인 음영을 만듭니다.
        try:
            grad_x = ndimage.sobel(depth_map, axis=1) / 8.0
            grad_y = ndimage.sobel(depth_map, axis=0) / 8.0
        except Exception:
            grad_y, grad_x = np.gradient(depth_map)
        
        # 조명 방향
        light_rad = np.radians(light_angle)
        light_x = np.cos(light_rad)
        light_y = np.sin(light_rad)
        
        # 램버시안 셰이딩
        # 법선 벡터: (-grad_x, -grad_y, 1) 정규화
        normal_z = np.ones_like(depth_map)
        norm = np.sqrt(grad_x**2 + grad_y**2 + normal_z**2)
        
        nx = -grad_x / norm
        ny = -grad_y / norm
        nz = normal_z / norm
        
        # 조명과 법선의 내적
        intensity = light_x * nx + light_y * ny + 0.5 * nz
        intensity = np.clip(intensity, 0, 1)
        
        # 탁본 효과: 높은 곳이 밝게
        image = (intensity * 255).astype(np.uint8)
        
        return image
    
    def _render_modern_rubbing(self, depth_map: np.ndarray,
                                light_angle: float) -> np.ndarray:
        """
        현대적인 탁본 스타일 (고대비)
        """
        base = self._render_traditional_rubbing(depth_map, light_angle)
        
        # 대비 증가
        enhanced = base.astype(np.float64)
        enhanced = (enhanced - enhanced.mean()) * 1.5 + enhanced.mean()
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _render_relief(self, depth_map: np.ndarray,
                       light_angle: float) -> np.ndarray:
        """
        릴리프(양각) 효과 렌더링
        """
        # Sobel 필터로 엣지 강조
        grad_x = ndimage.sobel(depth_map, axis=1)
        grad_y = ndimage.sobel(depth_map, axis=0)
        
        # 조명 방향에 따른 음영
        light_rad = np.radians(light_angle)
        emboss = np.cos(light_rad) * grad_x + np.sin(light_rad) * grad_y
        
        # 정규화
        emboss = (emboss - emboss.min()) / (emboss.max() - emboss.min() + 1e-10)
        image = (emboss * 255).astype(np.uint8)
        
        return image
    
    def generate_depth_map(self, flattened: FlattenedMesh,
                           width_pixels: int = 2000) -> RubbingImage:
        """
        깊이맵 이미지 생성
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비
            
        Returns:
            RubbingImage: 깊이맵 이미지
        """
        width_pixels = int(width_pixels)
        if width_pixels < 1:
            width_pixels = 1

        if flattened is None or getattr(flattened, "uv", None) is None or getattr(flattened, "faces", None) is None:
            raise ValueError("Flattened mesh is missing uv/faces.")
        if flattened.uv.ndim != 2 or flattened.uv.shape[0] == 0:
            raise ValueError("Flattened mesh has no UV vertices.")
        if flattened.faces.ndim != 2 or flattened.faces.shape[0] == 0:
            raise ValueError("Flattened mesh has no faces.")

        w_real = float(flattened.width)
        h_real = float(flattened.height)
        if not np.isfinite(w_real) or not np.isfinite(h_real) or w_real <= 1e-12:
            raise ValueError("Flattened mesh has invalid dimensions.")

        aspect_ratio = h_real / w_real
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        height_pixels = max(1, int(width_pixels * aspect_ratio))
        
        depth_map = self._create_depth_map(flattened, width_pixels, height_pixels)
        
        # 정규화 및 8비트 변환
        normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-10)
        image = (normalized * 255).astype(np.uint8)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )
    
    def generate_curvature_map(self, flattened: FlattenedMesh,
                               width_pixels: int = 2000) -> RubbingImage:
        """
        곡률맵 생성 (문양 강조)
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비
            
        Returns:
            RubbingImage: 곡률맵 이미지
        """
        width_pixels = int(width_pixels)
        if width_pixels < 1:
            width_pixels = 1

        if flattened is None or getattr(flattened, "uv", None) is None or getattr(flattened, "faces", None) is None:
            raise ValueError("Flattened mesh is missing uv/faces.")
        if flattened.uv.ndim != 2 or flattened.uv.shape[0] == 0:
            raise ValueError("Flattened mesh has no UV vertices.")
        if flattened.faces.ndim != 2 or flattened.faces.shape[0] == 0:
            raise ValueError("Flattened mesh has no faces.")

        w_real = float(flattened.width)
        h_real = float(flattened.height)
        if not np.isfinite(w_real) or not np.isfinite(h_real) or w_real <= 1e-12:
            raise ValueError("Flattened mesh has invalid dimensions.")

        aspect_ratio = h_real / w_real
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        height_pixels = max(1, int(width_pixels * aspect_ratio))
        
        depth_map = self._create_depth_map(flattened, width_pixels, height_pixels)
        
        # 라플라시안으로 곡률 계산
        curvature = ndimage.laplace(depth_map)
        
        # 정규화
        curvature = np.abs(curvature)
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-10)
        image = (curvature * 255).astype(np.uint8)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )


# 테스트용
if __name__ == '__main__':
    print("Surface Visualizer module loaded successfully")
    print("Use: visualizer = SurfaceVisualizer(); rubbing = visualizer.generate_rubbing(flattened)")
