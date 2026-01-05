import numpy as np
from dataclasses import dataclass, field
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Config
# -----------------------------
@dataclass
class CacheConfig:
    line_size: int = 64
    l1_bytes: int = 32 * 1024
    l2_bytes: int = 512 * 1024
    l3_bytes: int = 8 * 1024 * 1024


@dataclass
class TrafficConfig:
    # scalar bytes for "No Cache worst" accounting
    a_bytes: int = 2   # bf16
    w_bytes: int = 2   # bf16
    c_bytes: int = 2   # bf16 (モデルの出力をbf16で書く想定。fp32にしたければ4にする)

    flops_per_fma: int = 2


@dataclass
class VisConfig:
    img_path: str = "./assets/2.jpg"
    N: int = 100

    Ti: int = 25
    Tj: int = 25
    Tk: int = 25

    scale: int = 5 * 2
    gap: int = 120
    top_pad: int = 70
    bottom_pad: int = 120
    bg: tuple = (250, 250, 250)

    grid_width: int = 8
    tile_outline_width: int = 6
    point_r: int = 6

    duration_ms: int = 360
    out_gif: str = "./assets/matmul_tiling_cache_model.gif"
    out_preview: str = "./out_preview.png"

    seed: int = 0
    draw_every_fma: int = 2500 * 10

    # ここを default_factory に変更
    cache: CacheConfig = field(default_factory=CacheConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)


# -----------------------------
# Fonts
# -----------------------------
def load_fonts():
    # 環境依存なのでfallback多めにする
    candidates = [
        ("Arial Unicode.ttf", 30, 22, 18),
        ("DejaVuSans.ttf", 30, 22, 18),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 30, 22, 18),
    ]
    for path, s1, s2, s3 in candidates:
        try:
            font_big = ImageFont.truetype(path, s1)
            font_mid = ImageFont.truetype(path, s2)
            font_small = ImageFont.truetype(path, s3)
            return font_big, font_mid, font_small
        except:
            pass
    # 最後のfallback
    f = ImageFont.load_default()
    return f, f, f


def format_si(x: float, kind: str) -> str:
    if kind == "flop":
        units = ["FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP"]
    else:
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(x)
    k = 1000.0
    u = 0
    while abs(x) >= k and u < len(units) - 1:
        x /= k
        u += 1
    return f"{x:.3g} {units[u]}"


# -----------------------------
# Cache Model (inclusive, fully-assoc LRU)
# -----------------------------
class CacheLevel:
    def __init__(self, name: str, capacity_bytes: int, line_size: int, lower=None):
        self.name = name
        self.capacity_lines = max(1, capacity_bytes // line_size)
        self.line_size = line_size
        self.lower = lower
        self.lines = OrderedDict()  # line_id -> dirty(bool)

    def _evict_if_needed(self):
        br = bw = 0
        while len(self.lines) > self.capacity_lines:
            line_id, dirty = self.lines.popitem(last=False)  # LRU
            if dirty:
                if self.lower is None:
                    bw += self.line_size
                else:
                    br2, bw2 = self.lower.writeback(line_id)
                    br += br2
                    bw += bw2
        return br, bw

    def read(self, line_id: int):
        if line_id in self.lines:
            dirty = self.lines.pop(line_id)
            self.lines[line_id] = dirty
            return 0, 0

        br = bw = 0
        if self.lower is None:
            br += self.line_size
        else:
            br2, bw2 = self.lower.read(line_id)
            br += br2
            bw += bw2

        self.lines[line_id] = False
        br2, bw2 = self._evict_if_needed()
        return br + br2, bw + bw2

    def write(self, line_id: int):
        # write-allocate + write-back
        if line_id in self.lines:
            _dirty = self.lines.pop(line_id)
            self.lines[line_id] = True
            return 0, 0

        br = bw = 0
        # RFO: bring the line first
        if self.lower is None:
            br += self.line_size
        else:
            br2, bw2 = self.lower.read(line_id)
            br += br2
            bw += bw2

        self.lines[line_id] = True
        br2, bw2 = self._evict_if_needed()
        return br + br2, bw + bw2

    def writeback(self, line_id: int):
        # receive a dirty full line from upper, no RFO
        br = bw = 0
        if line_id in self.lines:
            _dirty = self.lines.pop(line_id)
            self.lines[line_id] = True
        else:
            self.lines[line_id] = True

        br2, bw2 = self._evict_if_needed()
        return br + br2, bw + bw2

    def flush(self):
        br = bw = 0
        for line_id, dirty in list(self.lines.items()):
            if dirty:
                if self.lower is None:
                    bw += self.line_size
                else:
                    br2, bw2 = self.lower.writeback(line_id)
                    br += br2
                    bw += bw2
        self.lines.clear()
        if self.lower is not None:
            br2, bw2 = self.lower.flush()
            br += br2
            bw += bw2
        return br, bw


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_tile_grid(draw: ImageDraw.ImageDraw, x0, y0, step_px, size_px, width=6, color=(0, 0, 0)):
    for t in range(0, size_px + 1, step_px):
        draw.line([(x0 + t, y0), (x0 + t, y0 + size_px)], width=width, fill=color)
        draw.line([(x0, y0 + t), (x0 + size_px, y0 + t)], width=width, fill=color)


def dot(odraw: ImageDraw.ImageDraw, x, y, r, fill, outline=(0, 0, 0, 230)):
    odraw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)


# -----------------------------
# Main
# -----------------------------
def main(cfg: VisConfig):
    font_big, font_mid, font_small = load_fonts()

    # ----- Load image (A)
    A_img = Image.open(cfg.img_path).convert("RGB").resize((cfg.N, cfg.N), Image.Resampling.LANCZOS)
    A_gray = np.asarray(A_img.convert("L"), dtype=np.float32) / 255.0

    # ----- Random weight (numeric + RGB vis)
    rng = np.random.default_rng(cfg.seed)
    W_num = (rng.standard_normal((cfg.N, cfg.N)).astype(np.float32)) * 0.25
    W_vis_rgb = rng.integers(0, 256, size=(cfg.N, cfg.N, 3), dtype=np.uint8)
    W_img = Image.fromarray(W_vis_rgb, mode="RGB")

    # Full output for consistent normalization
    C_full = A_gray @ W_num
    cmin, cmax = float(C_full.min()), float(C_full.max())
    if abs(cmax - cmin) < 1e-8:
        cmax = cmin + 1.0

    def to_vis_u8(mat):
        x = (mat - cmin) / (cmax - cmin)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    # ----- Layout
    panel = cfg.N * cfg.scale
    H = cfg.top_pad + panel + cfg.bottom_pad
    W_total = panel + cfg.gap + panel + cfg.gap + panel

    xA, yP = 0, cfg.top_pad
    xW = panel + cfg.gap
    xC = panel + cfg.gap + panel + cfg.gap

    A_panel_base = A_img.resize((panel, panel), Image.Resampling.NEAREST)
    W_panel_base = W_img.resize((panel, panel), Image.Resampling.NEAREST)

    # ----- Tiling ranges
    tiles_i = list(range(0, cfg.N, cfg.Ti))
    tiles_j = list(range(0, cfg.N, cfg.Tj))
    tiles_k = list(range(0, cfg.N, cfg.Tk))

    # ----- Cache hierarchy
    cc = cfg.cache
    l3 = CacheLevel("L3", cc.l3_bytes, cc.line_size, lower=None)
    l2 = CacheLevel("L2", cc.l2_bytes, cc.line_size, lower=l3)
    l1 = CacheLevel("L1", cc.l1_bytes, cc.line_size, lower=l2)

    # Address mapping (row-major). base offsets separate arrays.
    eb = cfg.traffic.a_bytes  # address stride uses element bytes (bf16想定)
    base_A = 0
    size_mat = cfg.N * cfg.N * eb
    base_W = base_A + size_mat + 4096
    base_C = base_W + size_mat + 4096

    def addr_A(i, k): return base_A + (i * cfg.N + k) * eb
    def addr_W(k, j): return base_W + (k * cfg.N + j) * eb
    def addr_C(i, j): return base_C + (i * cfg.N + j) * eb

    def line_id(addr): return addr // cc.line_size

    # ----- Accum + frames
    C_acc = np.zeros((cfg.N, cfg.N), dtype=np.float32)
    frames = []

    # Metrics
    total_fma = 0
    no_cache_bytes = 0  # scalar-byte worst
    hbm_read = 0
    hbm_write = 0

    # Utility: render a frame given current tile + a representative point
    def render_frame(i0, j0, k0, ii, jj, kk):
        nonlocal C_acc, total_fma, no_cache_bytes, hbm_read, hbm_write

        # C panel from current accumulation (tile-step granularity)
        C_img = Image.fromarray(to_vis_u8(C_acc), mode="L").convert("RGB")
        C_panel_base = C_img.resize((panel, panel), Image.Resampling.NEAREST)

        canvas = Image.new("RGB", (W_total, H), cfg.bg)
        draw = ImageDraw.Draw(canvas)
        canvas.paste(A_panel_base, (xA, yP))
        canvas.paste(W_panel_base, (xW, yP))
        canvas.paste(C_panel_base, (xC, yP))

        # Titles + operators
        draw.text((xA + 8, 10), "A: Input", font=font_mid, fill=(0, 0, 0))
        draw.text((xW + 8, 10), "W: Weight", font=font_mid, fill=(0, 0, 0))
        draw.text((xC + 8, 10), "C: Output", font=font_mid, fill=(0, 0, 0))
        draw.text((panel + cfg.gap // 2 - 12, yP + panel // 2 - 20), "@", font=font_big, fill=(0, 0, 0))
        draw.text((panel + cfg.gap + panel + cfg.gap // 2 - 12, yP + panel // 2 - 20), "=", font=font_big, fill=(0, 0, 0))

        # Tile grids
        draw_tile_grid(draw, xA, yP, cfg.Ti * cfg.scale, panel, width=cfg.grid_width)
        draw_tile_grid(draw, xW, yP, cfg.Tj * cfg.scale, panel, width=cfg.grid_width)
        draw_tile_grid(draw, xC, yP, cfg.Ti * cfg.scale, panel, width=cfg.grid_width)

        # Highlight tile rectangles (A read, W read, C tile)
        ax1 = xA + k0 * cfg.scale
        ay1 = yP + i0 * cfg.scale
        ax2 = xA + (k0 + cfg.Tk) * cfg.scale
        ay2 = yP + (i0 + cfg.Ti) * cfg.scale

        wx1 = xW + j0 * cfg.scale
        wy1 = yP + k0 * cfg.scale
        wx2 = xW + (j0 + cfg.Tj) * cfg.scale
        wy2 = yP + (k0 + cfg.Tk) * cfg.scale

        cx1 = xC + j0 * cfg.scale
        cy1 = yP + i0 * cfg.scale
        cx2 = xC + (j0 + cfg.Tj) * cfg.scale
        cy2 = yP + (i0 + cfg.Ti) * cfg.scale

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([ax1, ay1, ax2, ay2], fill=(255, 80, 80, 70), outline=(255, 0, 0, 220), width=cfg.tile_outline_width)
        od.rectangle([wx1, wy1, wx2, wy2], fill=(255, 80, 80, 70), outline=(255, 0, 0, 220), width=cfg.tile_outline_width)
        od.rectangle([cx1, cy1, cx2, cy2], fill=(80, 160, 255, 60), outline=(0, 90, 255, 220), width=cfg.tile_outline_width)

        # Point (A[i,k], W[k,j], C[i,j])
        pi = i0 + ii
        pj = j0 + jj
        pk = k0 + kk

        pxA = xA + pk * cfg.scale
        pyA = yP + pi * cfg.scale
        pxW = xW + pj * cfg.scale
        pyW = yP + pk * cfg.scale
        pxC = xC + pj * cfg.scale
        pyC = yP + pi * cfg.scale

        dot(od, pxA, pyA, cfg.point_r, fill=(255, 0, 0, 230))
        dot(od, pxW, pyW, cfg.point_r, fill=(255, 0, 0, 230))
        dot(od, pxC, pyC, cfg.point_r, fill=(0, 90, 255, 230))

        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        # Metrics display
        total_flop = total_fma * cfg.traffic.flops_per_fma

        # No-cache worst (scalar bytes)
        bf_no = no_cache_bytes / max(1.0, total_flop)

        # With-cache (HBM line traffic)
        hbm_total = hbm_read + hbm_write
        bf_ca = hbm_total / max(1.0, total_flop)

        tx = 12
        ty = cfg.top_pad + panel + 12

        # 5 lines: FLOP, NoCache traffic+B/F, Cache traffic+B/F
        draw.text((tx, ty + 0),  f"Total FLOP: {format_si(total_flop, 'flop')}", font=font_mid, fill=(0, 0, 0))
        draw.text((tx, ty + 28), f"Traffic (No Cache): {format_si(no_cache_bytes, 'byte')}", font=font_mid, fill=(0, 0, 0))
        draw.text((tx, ty + 56), f"B/F (No Cache): {bf_no:.3g}", font=font_mid, fill=(0, 0, 0))
        draw.text((tx, ty + 84), f"Traffic (With Cache): {format_si(hbm_total, 'byte')}", font=font_mid, fill=(0, 0, 0))
        draw.text((tx, ty + 112), f"B/F (With Cache): {bf_ca:.3g}", font=font_mid, fill=(0, 0, 0))

        return canvas

    # -----------------------------
    # Simulation:
    # - Outer loops are tiled GEMM order.
    # - We update numeric C_acc per (i0,j0,k0) step using NumPy.
    # - We simulate per-FMA traffic inside the tile, but render only every draw_every_fma FMAs.
    # -----------------------------
    # NOTE: C tile cache behavior:
    # - "With Cache" model: C tile is read once at start of (i0,j0), written once at end of (i0,j0).
    # - "No Cache worst": we still count C load+store per FMA (super pessimistic).
    for i0 in tiles_i:
        for j0 in tiles_j:
            # With-cache: bring C tile (read) once for this (i0,j0)
            for ii in range(cfg.Ti):
                for jj in range(cfg.Tj):
                    br, bw = l1.read(line_id(addr_C(i0 + ii, j0 + jj)))
                    hbm_read += br
                    hbm_write += bw

            for k0 in tiles_k:
                # (A_tile @ W_tile) numeric update at tile-step granularity
                C_acc[i0:i0+cfg.Ti, j0:j0+cfg.Tj] += (
                    A_gray[i0:i0+cfg.Ti, k0:k0+cfg.Tk] @ W_num[k0:k0+cfg.Tk, j0:j0+cfg.Tj]
                )

                # Per-FMA simulation inside this tile
                # Use "micro-kernel-ish" order for reuse: kk outer, then ii, then jj
                for kk in range(cfg.Tk):
                    for ii in range(cfg.Ti):
                        # Cache model: A[i,k] read
                        # (jj loop内で同じ A[i,k] を繰り返し使うので、キャッシュヒットが起きやすい形になる)
                        for jj in range(cfg.Tj):
                            # ---- Metrics update (1 FMA) ----
                            total_fma += 1

                            # No-cache worst: every FMA reloads everything from HBM (scalar bytes)
                            no_cache_bytes += (cfg.traffic.a_bytes + cfg.traffic.w_bytes +
                                               cfg.traffic.c_bytes + cfg.traffic.c_bytes)

                            # With-cache: simulate scalar accesses but counted as cache-line HBM traffic
                            # A load
                            br, bw = l1.read(line_id(addr_A(i0 + ii, k0 + kk)))
                            hbm_read += br
                            hbm_write += bw

                            # W load
                            br, bw = l1.read(line_id(addr_W(k0 + kk, j0 + jj)))
                            hbm_read += br
                            hbm_write += bw

                            # C access inside kernel is assumed register-resident in the cache model,
                            # so we DO NOT touch C here.

                            # Render occasionally
                            if (total_fma % cfg.draw_every_fma) == 0:
                                frame = render_frame(i0, j0, k0, ii, jj, kk)
                                frames.append(frame)

                                # Save a preview around the first render
                                if len(frames) == 1:
                                    frame.save(cfg.out_preview)

            # With-cache: write back C tile once for this (i0,j0)
            for ii in range(cfg.Ti):
                for jj in range(cfg.Tj):
                    br, bw = l1.write(line_id(addr_C(i0 + ii, j0 + jj)))
                    hbm_read += br
                    hbm_write += bw

    # Flush caches (write back dirty lines)
    br, bw = l1.flush()
    hbm_read += br
    hbm_write += bw

    # Ensure we have at least 1 frame
    if not frames:
        frames.append(render_frame(tiles_i[0], tiles_j[0], tiles_k[0], 0, 0, 0))
        frames[0].save(cfg.out_preview)

    # Save GIF
    # PILのADAPTIVEは convert の palette 引数に Image.ADAPTIVE を渡すのが安定
    frames_p = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    frames_p[0].save(
        cfg.out_gif,
        save_all=True,
        append_images=frames_p[1:],
        duration=cfg.duration_ms,
        loop=0,
        optimize=False,
    )

    total_flop = total_fma * cfg.traffic.flops_per_fma
    print("Wrote:", cfg.out_gif)
    print("Preview:", cfg.out_preview)
    print("Total FMA:", total_fma)
    print("Total FLOP:", total_flop)
    print("Traffic (No Cache worst, scalar bytes):", no_cache_bytes)
    print("B/F (No Cache worst):", no_cache_bytes / total_flop)
    print("HBM read/write (With Cache, line model):", hbm_read, hbm_write)
    print("Traffic (With Cache):", hbm_read + hbm_write)
    print("B/F (With Cache):", (hbm_read + hbm_write) / total_flop)


if __name__ == "__main__":
    cfg = VisConfig()
    main(cfg)
