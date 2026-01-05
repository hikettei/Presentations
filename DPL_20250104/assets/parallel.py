import os
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    img_path: str = "./assets/2.jpg"   # 100x100 にリサイズして使う
    out_dir: str = "./assets"

    N: int = 100
    tile: int = 10          # 10x10 tiles

    # Viz sizing
    scale: int = 18          # N*scale がパネルサイズ
    top_pad: int = 70
    bottom_pad: int = 90
    gap: int = 90
    bg: Tuple[int, int, int] = (250, 250, 250)

    grid_width: int = 6
    tile_outline_width: int = 6
    point_r: int = 5

    # CPU parallel
    cpu_threads: int = 4

    # GPU parallel (CTA scheduling view)
    gpu_cta_per_step: int = 8          # 1 step で同時に走ってる block 数のイメージ
    gpu_substeps: int = 6              # 1 step あたり何フレームで thread の動きを見せるか
    gpu_block_threads_1d: int = 10     # 10x10 threads をイメージ（tile と合わせる）

    # SIMD
    simd_width: int = 10
    simd_rows: int = 4                # 何行ぶんアニメするか（大きいほど長くなる）

    # GIF
    duration_ms: int = 160            # 再生速度
    optimize: bool = False            # True は遅くなることがある


# -----------------------------
# Fonts / helpers
# -----------------------------
def load_fonts():
    # 環境依存なので，取れなければ default に落とす
    candidates = [
        ("Arial Unicode.ttf", 30, 22, 18),
        ("DejaVuSans.ttf", 30, 22, 18),
    ]
    for name, s_big, s_mid, s_small in candidates:
        try:
            font_big = ImageFont.truetype(name, s_big)
            font_mid = ImageFont.truetype(name, s_mid)
            font_small = ImageFont.truetype(name, s_small)
            return font_big, font_mid, font_small
        except Exception:
            pass
    f = ImageFont.load_default()
    return f, f, f


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_tile_grid(draw: ImageDraw.ImageDraw, x0, y0, step_px, size_px, width=6, color=(0, 0, 0)):
    for t in range(0, size_px + 1, step_px):
        draw.line([(x0 + t, y0), (x0 + t, y0 + size_px)], width=width, fill=color)
        draw.line([(x0, y0 + t), (x0 + size_px, y0 + t)], width=width, fill=color)


def rgba_overlay(base_rgb: Image.Image, painter_fn):
    overlay = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    painter_fn(od)
    return Image.alpha_composite(base_rgb.convert("RGBA"), overlay).convert("RGB")


def to_palette_gif(frames: List[Image.Image], out_path: str, duration_ms: int, optimize: bool):
    # P化してサイズ削減（巨大解像度だとそれでも重いので，frame数を抑えるのが本質）
    frames_p = [f.convert("P", palette=Image.Palette.ADAPTIVE) for f in frames]
    frames_p[0].save(
        out_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=0,
        optimize=optimize,
    )


def panel_from_image(img: Image.Image, panel: int) -> Image.Image:
    return img.resize((panel, panel), Image.Resampling.NEAREST)


# -----------------------------
# Base canvas builder
# -----------------------------
def make_base(cfg: Config):
    font_big, font_mid, font_small = load_fonts()

    # Load image as "iteration space"
    A_img = Image.open(cfg.img_path).convert("RGB").resize((cfg.N, cfg.N), Image.Resampling.LANCZOS)

    panel = cfg.N * cfg.scale
    W_total = panel
    H = cfg.top_pad + panel + cfg.bottom_pad

    base_panel = panel_from_image(A_img, panel)

    def blank_canvas(title: str) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        canvas = Image.new("RGB", (W_total, H), cfg.bg)
        draw = ImageDraw.Draw(canvas)
        canvas.paste(base_panel, (0, cfg.top_pad))

        draw.text((10, 10), title, font=font_mid, fill=(0, 0, 0))

        # grid
        step_px = cfg.tile * cfg.scale
        draw_tile_grid(draw, 0, cfg.top_pad, step_px, panel, width=cfg.grid_width)

        return canvas, draw

    return A_img, base_panel, panel, H, W_total, (font_big, font_mid, font_small), blank_canvas


# -----------------------------
# (A) CPU Parallel GIF
# -----------------------------
def make_parallel_cpu_gif(cfg: Config, out_path: str):
    _, base_panel, panel, H, W_total, (font_big, font_mid, font_small), blank_canvas = make_base(cfg)

    tiles = []
    for i0 in range(0, cfg.N, cfg.tile):
        for j0 in range(0, cfg.N, cfg.tile):
            tiles.append((i0, j0))
    n_tiles = len(tiles)
    steps = math.ceil(n_tiles / cfg.cpu_threads)

    # visually distinct thread colors
    thread_colors = [
        (255, 80, 80),
        (80, 160, 255),
        (80, 220, 120),
        (255, 180, 60),
        (180, 120, 255),
        (0, 200, 200),
    ]

    done = set()
    frames: List[Image.Image] = []

    for step in range(steps):
        active = []
        for t in range(cfg.cpu_threads):
            idx = step * cfg.cpu_threads + t
            if idx < n_tiles:
                active.append((t, tiles[idx]))
        # Mark active as done for next steps（見た目上の進捗）
        for _, (i0, j0) in active:
            done.add((i0, j0))

        canvas, draw = blank_canvas("CPU Parallel: tiles split across threads")
        canvas.paste(base_panel, (0, cfg.top_pad))

        def paint(od: ImageDraw.ImageDraw):
            # done tiles: light tint
            for (i0, j0) in done:
                x1 = j0 * cfg.scale
                y1 = cfg.top_pad + i0 * cfg.scale
                x2 = (j0 + cfg.tile) * cfg.scale
                y2 = cfg.top_pad + (i0 + cfg.tile) * cfg.scale
                od.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 18))

            # active tiles: strong tint + label
            for t, (i0, j0) in active:
                c = thread_colors[t % len(thread_colors)]
                x1 = j0 * cfg.scale
                y1 = cfg.top_pad + i0 * cfg.scale
                x2 = (j0 + cfg.tile) * cfg.scale
                y2 = cfg.top_pad + (i0 + cfg.tile) * cfg.scale
                od.rectangle([x1, y1, x2, y2], fill=(c[0], c[1], c[2], 80), outline=(0, 0, 0, 220), width=cfg.tile_outline_width)

        canvas = rgba_overlay(canvas, paint)
        draw = ImageDraw.Draw(canvas)

        # bottom text（スライド用に短く）
        y0 = cfg.top_pad + panel + 14
        draw.text((10, y0 + 0),  f"Threads: {cfg.cpu_threads}", font=font_mid, fill=(0, 0, 0))
        draw.text((10, y0 + 26), f"Step: {step+1}/{steps}", font=font_mid, fill=(0, 0, 0))

        # legend
        lx = 10
        ly = y0 + 56
        for t in range(cfg.cpu_threads):
            c = thread_colors[t % len(thread_colors)]
            draw.rectangle([lx, ly + 4, lx + 24, ly + 24], fill=c, outline=(0, 0, 0))
            draw.text((lx + 34, ly), f"T{t}", font=font_small, fill=(0, 0, 0))
            lx += 90

        frames.append(canvas)

    to_palette_gif(frames, out_path, cfg.duration_ms, cfg.optimize)


# -----------------------------
# (B) GPU Parallel GIF
# -----------------------------
def make_parallel_gpu_gif(cfg: Config, out_path: str):
    _, base_panel, panel, H, W_total, (font_big, font_mid, font_small), blank_canvas = make_base(cfg)

    # CTA = one tile (i0, j0)
    blocks = []
    for by in range(0, cfg.N, cfg.tile):
        for bx in range(0, cfg.N, cfg.tile):
            blocks.append((by, bx))
    n_blocks = len(blocks)
    steps = math.ceil(n_blocks / cfg.gpu_cta_per_step)

    block_colors = [
        (255, 80, 80),
        (80, 160, 255),
        (80, 220, 120),
        (255, 180, 60),
        (180, 120, 255),
        (0, 200, 200),
        (255, 120, 200),
        (160, 160, 160),
    ]

    frames: List[Image.Image] = []
    done = set()

    for step in range(steps):
        # active CTAs this step
        active = []
        for s in range(cfg.gpu_cta_per_step):
            idx = step * cfg.gpu_cta_per_step + s
            if idx < n_blocks:
                active.append((s, blocks[idx]))
        for _, b in active:
            done.add(b)

        # within-step animation to show "threads moving inside a block"
        for sub in range(cfg.gpu_substeps):
            canvas, draw = blank_canvas("GPU Parallel: blocks + threads (conceptual)")
            canvas.paste(base_panel, (0, cfg.top_pad))

            def paint(od: ImageDraw.ImageDraw):
                # done blocks: slight tint
                for (i0, j0) in done:
                    x1 = j0 * cfg.scale
                    y1 = cfg.top_pad + i0 * cfg.scale
                    x2 = (j0 + cfg.tile) * cfg.scale
                    y2 = cfg.top_pad + (i0 + cfg.tile) * cfg.scale
                    od.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 14))

                # active blocks: strong tint + outline
                for s, (i0, j0) in active:
                    c = block_colors[s % len(block_colors)]
                    x1 = j0 * cfg.scale
                    y1 = cfg.top_pad + i0 * cfg.scale
                    x2 = (j0 + cfg.tile) * cfg.scale
                    y2 = cfg.top_pad + (i0 + cfg.tile) * cfg.scale
                    od.rectangle([x1, y1, x2, y2], fill=(c[0], c[1], c[2], 70), outline=(0, 0, 0, 220), width=cfg.tile_outline_width)

                    # show a moving "thread" dot inside each CTA
                    # map substep to (tx, ty) in 10x10 thread grid
                    tx = (sub + 2 * s) % cfg.gpu_block_threads_1d
                    ty = (2 * sub + s) % cfg.gpu_block_threads_1d
                    px = x1 + (tx + 0.5) * (cfg.scale)   # each "thread" -> 1 pixel in tile space, scaled
                    py = y1 + (ty + 0.5) * (cfg.scale)
                    r = cfg.point_r
                    od.ellipse([px - r, py - r, px + r, py + r], fill=(0, 0, 0, 230))

            canvas = rgba_overlay(canvas, paint)
            draw = ImageDraw.Draw(canvas)

            y0 = cfg.top_pad + panel + 14
            draw.text((10, y0 + 0),  f"CTAs per step: {cfg.gpu_cta_per_step}", font=font_mid, fill=(0, 0, 0))
            draw.text((10, y0 + 26), f"Step: {step+1}/{steps}", font=font_mid, fill=(0, 0, 0))

            frames.append(canvas)

    to_palette_gif(frames, out_path, cfg.duration_ms, cfg.optimize)


# -----------------------------
# (C) SIMD / Strip-mine GIF
# -----------------------------
def make_simd_gif(cfg: Config, out_path: str):
    _, base_panel, panel, H, W_total, (font_big, font_mid, font_small), blank_canvas = make_base(cfg)

    # 右側に "SIMD register view" パネルを追加する
    reg_gap = cfg.gap
    lane_box_w = 150
    lane_box_h = 110
    lane_box_gap_y = 26
    reg_panel_w = lane_box_w + 40
    W_total2 = panel + reg_gap + reg_panel_w

    # 見やすい lane 配色（大袈裟に）
    lane_colors = [
        (255, 80, 80),    # lane0
        (80, 160, 255),   # lane1
        (80, 220, 120),   # lane2
        (255, 180, 60),   # lane3
    ]

    frames: List[Image.Image] = []

    # Animate a few rows, and within each row animate j_outer
    rows = list(range(cfg.simd_rows))
    groups_per_row = cfg.N // cfg.simd_width

    for rr in rows:
        i = rr  # fixed row
        for j_outer in range(groups_per_row):
            j0 = j_outer * cfg.simd_width

            # --- canvas (wider) ---
            canvas = Image.new("RGB", (W_total2, H), cfg.bg)
            draw = ImageDraw.Draw(canvas)

            # left: image panel
            canvas.paste(base_panel, (0, cfg.top_pad))

            # title
            draw.text((10, 10), "SIMD / Strip-mine: lanes are exaggerated", font=font_mid, fill=(0, 0, 0))

            # grid
            step_px = cfg.tile * cfg.scale
            draw_tile_grid(draw, 0, cfg.top_pad, step_px, panel, width=cfg.grid_width)

            # right panel origin
            rx0 = panel + reg_gap
            ry0 = cfg.top_pad

            # right panel header
            draw.text((rx0 + 10, 10), "YMM lanes (conceptual)", font=font_mid, fill=(0, 0, 0))

            # positions for lane boxes
            boxes = []
            cur_y = ry0 + 10
            for lane in range(cfg.simd_width):
                bx1 = rx0 + 20
                by1 = cur_y
                bx2 = bx1 + lane_box_w
                by2 = by1 + lane_box_h
                boxes.append((bx1, by1, bx2, by2))
                cur_y += lane_box_h + lane_box_gap_y

            # paint overlays
            def paint(od: ImageDraw.ImageDraw):
                # highlight the whole active row lightly (thicker band: 3 rows分に拡張して視認性UP)
                band_h = 3  # 1→3 にして，横帯を太くする
                y1 = cfg.top_pad + max(0, i - 1) * cfg.scale
                y2 = cfg.top_pad + min(cfg.N, i - 1 + band_h) * cfg.scale
                od.rectangle([0, y1, panel, y2], fill=(0, 0, 0, 18))

                # highlight SIMD vector chunk strongly (太い枠)
                x1 = j0 * cfg.scale
                x2 = (j0 + cfg.simd_width) * cfg.scale
                od.rectangle(
                    [x1, cfg.top_pad + i * cfg.scale, x2, cfg.top_pad + (i + 1) * cfg.scale],
                    fill=(0, 0, 0, 0),
                    outline=(0, 0, 0, 255),
                    width=max(8, cfg.tile_outline_width),
                )

                # per-lane exaggerated markers: 太い縦バー＋大きい点＋laneラベル
                for lane in range(cfg.simd_width):
                    c = lane_colors[lane % len(lane_colors)]
                    j = j0 + lane

                    # element center in left panel
                    ex = (j + 0.5) * cfg.scale
                    ey = cfg.top_pad + (i + 0.5) * cfg.scale

                    # draw a thick vertical bar around that element (誇張)
                    bar_w = max(10, cfg.scale // 2 + 8)
                    od.rectangle(
                        [ex - bar_w, cfg.top_pad + i * cfg.scale - 14, ex + bar_w, cfg.top_pad + (i + 1) * cfg.scale + 14],
                        fill=(c[0], c[1], c[2], 80),
                        outline=(0, 0, 0, 230),
                        width=4,
                    )

                    # big dot
                    r = max(10, cfg.point_r + 6)
                    od.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(c[0], c[1], c[2], 230), outline=(0, 0, 0, 230), width=3)

                    # lane label near the dot
                    # （PILのtextはRGBA overlayでも描けるが，ここでは枠だけにして後でdrawで文字を載せる）
                    # connection line to lane box
                    bx1, by1, bx2, by2 = boxes[lane]
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    od.line([(ex, ey), (cx, cy)], fill=(0, 0, 0, 200), width=4)

                # right lane boxes
                for lane in range(cfg.simd_width):
                    c = lane_colors[lane % len(lane_colors)]
                    bx1, by1, bx2, by2 = boxes[lane]
                    od.rectangle([bx1, by1, bx2, by2], fill=(c[0], c[1], c[2], 90), outline=(0, 0, 0, 255), width=6)

            canvas = rgba_overlay(canvas, paint)
            draw = ImageDraw.Draw(canvas)

            # put lane labels (big) and current mapping
            for lane in range(cfg.simd_width):
                bx1, by1, bx2, by2 = boxes[lane]
                j = j0 + lane
                draw.text((bx1 + 12, by1 + 10), f"lane {lane}", font=font_mid, fill=(0, 0, 0))
                draw.text((bx1 + 12, by1 + 42), f"A[{i},{j}]", font=font_mid, fill=(0, 0, 0))

            # bottom annotation (短く，スライド向け)
            yb = cfg.top_pad + panel + 14
            draw.text((10, yb + 0),  f"i = {i}", font=font_mid, fill=(0, 0, 0))
            draw.text((10, yb + 26), f"j_outer = {j_outer}  ,  lanes = 0..{cfg.simd_width-1}", font=font_mid, fill=(0, 0, 0))
            draw.text((10, yb + 52), f"j = j_outer*{cfg.simd_width} + lane", font=font_mid, fill=(0, 0, 0))

            frames.append(canvas)

    to_palette_gif(frames, out_path, cfg.duration_ms, cfg.optimize)

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "gpu", "simd", "all"], default="all")
    ap.add_argument("--img", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.img is not None:
        cfg.img_path = args.img
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    ensure_dir(cfg.out_dir)

    if args.mode in ("cpu", "all"):
        out = os.path.join(cfg.out_dir, "parallel_cpu.gif")
        make_parallel_cpu_gif(cfg, out)
        print("Wrote:", out)

    if args.mode in ("gpu", "all"):
        out = os.path.join(cfg.out_dir, "parallel_gpu.gif")
        make_parallel_gpu_gif(cfg, out)
        print("Wrote:", out)

    if args.mode in ("simd", "all"):
        out = os.path.join(cfg.out_dir, "simd_stripmine.gif")
        make_simd_gif(cfg, out)
        print("Wrote:", out)


if __name__ == "__main__":
    main()
