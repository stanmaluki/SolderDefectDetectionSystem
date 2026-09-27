"""Procedural Synthetic Solder Joint Generator for SolSight.

Generates:
1. Golden reference (defect-free) PCBA solder joint patches across discrete native resolutions.
2. Defect variants across 4 defect classes:
   - voids: dark irregular / elliptical hollows within the solder fillet.
   - bridging: high-intensity solder protrusion / bridge connecting outward.
   - cold_joints: rough, granular, matte texture with dispersed/weak specular reflection.
   - solder_amount: insufficient (underfill / shrunken) or excess (bulging overfilled solder).

Key design features:
- Natively rendered at requested target_size (e.g. 16, 32, 64, 128px) — NO downsampling from 128px.
- Fully parametric: geometry, pad size, specular highlight, substrate color, lighting direction.
- Outputs saved as PNG files normalized to [0, 255] RGB.
"""

import math
import random
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_base_pcb(size: int, rng: random.Random) -> np.ndarray:
    """Create a realistic PCBA substrate background with solder mask texture."""
    # Substrate hue: dark green with slight random variation
    r = rng.randint(15, 35)
    g = rng.randint(45, 80)
    b = rng.randint(20, 45)
    base_color = np.array([r, g, b], dtype=np.float32)

    # Substrate noise texture
    seed = rng.randint(0, 2**31 - 1)
    np_rng = np.random.default_rng(seed)
    noise = np_rng.uniform(-4.0, 4.0, (size, size, 3)).astype(np.float32)
    img = np.clip(base_color + noise, 0, 255)
    return img


def get_pixel_coords(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D (x, y) float32 coordinates for a size x size patch."""
    coords = np.arange(size, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    return x, y


def render_pad(
    img: np.ndarray,
    size: int,
    pad_radius: float,
    center: Tuple[float, float],
    rng: random.Random,
) -> np.ndarray:
    """Render the circular/rectangular copper/tin pad on the substrate."""
    cx, cy = center
    x, y = get_pixel_coords(size)
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2

    # Copper / tin pad color (golden-copper or tin-silver finish)
    is_hasl = rng.random() > 0.4
    if is_hasl:
        # Silver / tin pad
        pad_color = np.array([160 + rng.uniform(-10, 10), 165 + rng.uniform(-10, 10), 175 + rng.uniform(-10, 10)])
    else:
        # ENIG (electroless nickel immersion gold)
        pad_color = np.array([190 + rng.uniform(-15, 15), 160 + rng.uniform(-15, 15), 60 + rng.uniform(-10, 10)])

    # Antialiased pad edge
    edge_width = max(0.75, size / 32.0)
    pad_mask = np.clip((pad_radius - np.sqrt(dist_sq)) / edge_width, 0.0, 1.0)
    pad_mask = pad_mask[..., np.newaxis]

    img = img * (1.0 - pad_mask) + pad_color * pad_mask
    return img


def render_normal_joint(
    size: int,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Procedurally render a golden-reference (defect-free) solder joint patch.

    Natively computes shading, dome profile, and specular reflection at target_size.
    """
    if rng is None:
        rng = random.Random()

    cx = size / 2.0 + rng.uniform(-0.04 * size, 0.04 * size)
    cy = size / 2.0 + rng.uniform(-0.04 * size, 0.04 * size)
    radius = rng.uniform(0.32 * size, 0.42 * size)

    # 1. Base substrate
    img = create_base_pcb(size, rng)

    # 2. Pad
    pad_radius = radius * rng.uniform(1.15, 1.30)
    img = render_pad(img, size, pad_radius, (cx, cy), rng)

    # 3. Parametric 3D solder dome geometry
    x, y = get_pixel_coords(size)
    dx = (x - cx) / radius
    dy = (y - cy) / radius
    r_sq = dx ** 2 + dy ** 2

    fillet_mask = r_sq <= 1.0

    # Spherical / paraboloid surface normal calculation
    z = np.zeros_like(r_sq)
    z[fillet_mask] = np.sqrt(np.maximum(0.0, 1.0 - r_sq[fillet_mask]))

    # Surface normals (nx, ny, nz)
    nx = np.zeros_like(r_sq)
    ny = np.zeros_like(r_sq)
    nz = np.ones_like(r_sq)
    nx[fillet_mask] = dx[fillet_mask]
    ny[fillet_mask] = dy[fillet_mask]
    nz[fillet_mask] = z[fillet_mask]
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    norm = np.maximum(norm, 1e-6)
    nx /= norm
    ny /= norm
    nz /= norm

    # Light direction (angled specular reflection typical of AOI ring light)
    light_azimuth = rng.uniform(0, 2 * math.pi)
    light_elevation = rng.uniform(0.5, 0.9)  # Steep oblique lighting
    lx = math.cos(light_azimuth) * math.sqrt(1 - light_elevation ** 2)
    ly = math.sin(light_azimuth) * math.sqrt(1 - light_elevation ** 2)
    lz = light_elevation

    # Diffuse shading (Lambertian)
    diffuse = np.maximum(0.0, nx * lx + ny * ly + nz * lz)

    # Specular shading (Blinn-Phong)
    vx, vy, vz = 0.0, 0.0, 1.0  # View direction along Z (top-down camera)
    hx = lx + vx
    hy = ly + vy
    hz = lz + vz
    h_norm = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    hx /= h_norm
    hy /= h_norm
    hz /= h_norm

    shininess = rng.uniform(18.0, 36.0)
    specular = np.maximum(0.0, nx * hx + ny * hy + nz * hz) ** shininess

    # Solder material color (shiny silver metallic)
    base_solder_val = rng.uniform(120.0, 155.0)
    solder_rgb = np.array([base_solder_val - 2, base_solder_val, base_solder_val + 6], dtype=np.float32)

    # Composite solder appearance
    ambient = 0.35
    illum = ambient + 0.45 * diffuse + 0.65 * specular
    illum = illum[..., np.newaxis]

    solder_pixels = np.clip(solder_rgb * illum, 0.0, 255.0)

    # Subtle metallic micro-texture
    seed_solder = rng.randint(0, 2**31 - 1)
    solder_noise = np.random.default_rng(seed_solder).uniform(-3.0, 3.0, (size, size, 3))
    solder_pixels = np.clip(solder_pixels + solder_noise, 0.0, 255.0)

    # Antialiased fillet edge
    edge_width = max(0.75, size / 32.0)
    fillet_alpha = np.clip((radius - np.sqrt((x - cx) ** 2 + (y - cy) ** 2)) / edge_width, 0.0, 1.0)
    fillet_alpha = fillet_alpha[..., np.newaxis]

    final_img = img * (1.0 - fillet_alpha) + solder_pixels * fillet_alpha
    final_uint8 = np.clip(final_img, 0, 255).astype(np.uint8)

    return Image.fromarray(final_uint8)


def render_defect_void(
    size: int,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Render solder joint with dark void(s) / blowhole anomalies."""
    if rng is None:
        rng = random.Random()

    img = render_normal_joint(size, rng)
    draw = ImageDraw.Draw(img)

    cx = size / 2.0
    cy = size / 2.0
    fillet_r = 0.35 * size

    # Number of voids: 1 to 3
    num_voids = rng.randint(1, 3)
    for _ in range(num_voids):
        # Void position within fillet
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.1, 0.7) * fillet_r
        vx = cx + dist * math.cos(angle)
        vy = cy + dist * math.sin(angle)

        # Void radius: between ~5% and ~18% of patch size
        vr = rng.uniform(max(1.0, 0.05 * size), 0.18 * size)

        # Dark hollow color with uneven edge
        void_color = (
            rng.randint(20, 45),
            rng.randint(20, 45),
            rng.randint(25, 50),
        )
        bbox = [vx - vr, vy - vr, vx + vr, vy + vr]
        draw.ellipse(bbox, fill=void_color)

    return img


def render_defect_bridging(
    size: int,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Render solder joint with unwanted bridging protrusion to adjacent track/pad."""
    if rng is None:
        rng = random.Random()

    img = render_normal_joint(size, rng)
    draw = ImageDraw.Draw(img)

    cx = size / 2.0
    cy = size / 2.0

    # Bridge direction extending out of the patch boundary
    angle = rng.uniform(0, 2 * math.pi)
    bridge_width = rng.uniform(max(2.0, 0.12 * size), 0.28 * size)

    # Start inside the fillet, extend to the border
    x0 = cx + 0.2 * size * math.cos(angle)
    y0 = cy + 0.2 * size * math.sin(angle)
    x1 = cx + 0.65 * size * math.cos(angle)
    y1 = cy + 0.65 * size * math.sin(angle)

    bridge_color = (
        rng.randint(160, 210),
        rng.randint(165, 215),
        rng.randint(170, 225),
    )
    draw.line([x0, y0, x1, y1], fill=bridge_color, width=int(round(bridge_width)))

    return img


def render_defect_cold_joint(
    size: int,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Render cold/disturbed solder joint: matte, rough granular texture, weak/no specular."""
    if rng is None:
        rng = random.Random()

    base = render_normal_joint(size, rng)
    arr = np.array(base, dtype=np.float32)

    cx = size / 2.0
    cy = size / 2.0
    fillet_r = 0.38 * size

    x, y = get_pixel_coords(size)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = np.clip((fillet_r - dist) / max(1.0, size / 16.0), 0.0, 1.0)[..., np.newaxis]

    # Dim specular highlights (cold joints lack bright specular reflections)
    # and add strong high-frequency grain / crystalline roughness
    seed_rough = rng.randint(0, 2**31 - 1)
    roughness = np.random.default_rng(seed_rough).uniform(-28.0, 28.0, arr.shape)
    # Dull graying effect
    dulled = arr * 0.72 + 35.0 + roughness

    arr = arr * (1.0 - mask) + dulled * mask
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def render_defect_solder_amount(
    size: int,
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Render insufficient (underfill/pinhole) or excess (overflowing ball) solder amount."""
    if rng is None:
        rng = random.Random()

    is_excess = rng.random() > 0.5
    img = create_base_pcb(size, rng)

    cx = size / 2.0
    cy = size / 2.0

    if is_excess:
        # Massive bulging solder ball that overflows the pad
        radius = rng.uniform(0.44 * size, 0.49 * size)
        pad_radius = radius * 0.95  # Pad hidden or overwhelmed
    else:
        # Insufficient solder: tiny shrunken fillet, copper pin/pad exposed
        radius = rng.uniform(0.12 * size, 0.20 * size)
        pad_radius = 0.40 * size

    img = render_pad(img, size, pad_radius, (cx, cy), rng)

    x, y = get_pixel_coords(size)
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    fillet_mask = dist_sq <= (radius ** 2)

    solder_val = rng.uniform(120.0, 160.0)
    solder_rgb = np.array([solder_val, solder_val + 2, solder_val + 8], dtype=np.float32)

    # Simplified shading
    edge_width = max(0.75, size / 32.0)
    fillet_alpha = np.clip((radius - np.sqrt(dist_sq)) / edge_width, 0.0, 1.0)[..., np.newaxis]

    final_img = img * (1.0 - fillet_alpha) + solder_rgb * fillet_alpha
    return Image.fromarray(np.clip(final_img, 0, 255).astype(np.uint8))


def generate_dataset_split(
    target_sizes=(16, 32, 64, 128),
    train_count_per_tier: int = 500,
    val_normal_per_tier: int = 50,
    val_defect_per_class: int = 25,
    seed: int = 42,
    base_dir: str = "data/synthetic",
) -> None:
    """Generate the full procedural dataset according to the Rev 8 volume budget."""
    rng = random.Random(seed)
    base = Path(base_dir)

    print("=" * 80)
    print("Generating Synthetic Solder Joint Dataset (Discrete Native Tiers)")
    print("=" * 80)

    # 1. Training normals: 16px, 64px, 128px (32px is untrained!)
    train_tiers = [s for s in target_sizes if s != 32]
    for size in train_tiers:
        train_dir = base / f"train/{size}px"
        train_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {train_count_per_tier} train patches at native {size}px...")
        for i in range(train_count_per_tier):
            img = render_normal_joint(size, rng)
            img.save(train_dir / f"normal_{size}px_{i:04d}.png")

    # 2. Validation normals: 16px, 32px (untrained test), 64px, 128px
    for size in target_sizes:
        val_dir = base / f"val_normal/{size}px"
        val_dir.mkdir(parents=True, exist_ok=True)
        tier_label = " (untrained test)" if size == 32 else ""
        print(f"Generating {val_normal_per_tier} val normal patches at native {size}px{tier_label}...")
        for i in range(val_normal_per_tier):
            img = render_normal_joint(size, rng)
            img.save(val_dir / f"val_normal_{size}px_{i:04d}.png")

    # 3. Validation defects across 4 classes: 16px, 32px, 64px, 128px
    defect_renderers = {
        "voids": render_defect_void,
        "bridging": render_defect_bridging,
        "cold_joints": render_defect_cold_joint,
        "solder_amount": render_defect_solder_amount,
    }

    for dclass, renderer in defect_renderers.items():
        for size in target_sizes:
            defect_dir = base / f"val_defects/{dclass}/{size}px"
            defect_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generating {val_defect_per_class} {dclass} patches at native {size}px...")
            for i in range(val_defect_per_class):
                img = renderer(size, rng)
                img.save(defect_dir / f"{dclass}_{size}px_{i:04d}.png")

    print("=" * 80)
    print("Dataset procedural generation complete.")
    print("=" * 80)


if __name__ == "__main__":
    generate_dataset_split()
