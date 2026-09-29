"""Build arena300.wad: basic.wad geometry + 300-round ammo stockpile.

Reproducible UDMF edit of ViZDoom's bundled basic.wad (PWAD, MAP01):
  - 6x ClipBox (DoomEd 2048, 50 bullets each = 300 rounds) ringed around the
    player start for genuine engine pickup (ITEMCOUNT-verified, never conjured).
  - 3x Zombieman (DoomEd 3004) at fixed bearings for extra live targets.
All other lumps (geometry, nodes, ACS spawn/reward scripts) are byte-identical
to basic.wad, so physics, reward (-5/shot, +106/kill), and KILLCOUNT behave
identically. THINGS edits never invalidate the BSP.

Usage: python benchmarks/build_arena300.py [--out anydecision/games/data/arena300.wad]
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import vizdoom as vzd

PLAYER_START = (-384.0, 32.0)
CLIPBOX = 2048  # Box of bullets, 50 rounds
ZOMBIEMAN = 3004  # 20 HP skirmisher
# Full spawn-flag set mirrored from the player start: all skills, all game
# modes (single/dm/coop), all player classes. Absent mode flags keep things
# from spawning, which is why the first arena revision loaded with zero items.
SPAWN_FLAGS = """skill1 = true; skill2 = true; skill3 = true; skill4 = true;
  skill5 = true; skill6 = true; skill7 = true; skill8 = true;
  single = true; dm = true; coop = true;
  class1 = true; class2 = true; class3 = true; class4 = true;
  class5 = true; class6 = true; class7 = true; class8 = true;"""


def _thing(x: float, y: float, type_id: int, angle: int = 0) -> str:
    return (
        "\nthing\n{\n"
        f"x = {x:.3f};\n"
        f"y = {y:.3f};\n"
        f"angle = {angle};\n"
        f"type = {type_id};\n"
        f"{SPAWN_FLAGS}\n"
        "}\n"
    )


def build(base_wad: Path) -> bytes:
    raw = base_wad.read_bytes()
    magic, n_lumps, dir_off = struct.unpack("<4sII", raw[:12])
    assert magic == b"PWAD", f"expected PWAD base, got {magic}"
    entries = []
    for i in range(n_lumps):
        pos, size, name = struct.unpack("<II8s", raw[dir_off + i * 16:dir_off + (i + 1) * 16])
        entries.append((pos, size, name.rstrip(b"\x00").decode("latin-1")))
    lumps = {name: raw[pos:pos + size] for pos, size, name in entries}
    assert "TEXTMAP" in lumps, "base WAD is not UDMF"

    px, py = PLAYER_START
    text = lumps["TEXTMAP"].decode("utf-8")
    # 6 ClipBoxes on a 96-unit ring around the player start (300 rounds).
    for k in range(6):
        ang = math.radians(60.0 * k + 30.0)
        text += _thing(px + 96.0 * math.cos(ang), py + 96.0 * math.sin(ang), CLIPBOX)
    # 3 Zombiemen on a 288-unit ring (fixed bearings, deterministic map).
    for k, deg in enumerate((30.0, 150.0, 270.0)):
        ang = math.radians(deg)
        text += _thing(px + 288.0 * math.cos(ang), py + 288.0 * math.sin(ang), ZOMBIEMAN, angle=0)
    lumps["TEXTMAP"] = text.encode("utf-8")

    # Rebuild the WAD byte layout (lumps back-to-back, fresh directory).
    out = bytearray(b"PWAD" + struct.pack("<II", len(entries), 0))
    new_entries = []
    for _, _, name in entries:
        data = lumps[name]
        new_entries.append((len(out), len(data), name))
        out += data
    dir_off = len(out)
    out[8:12] = struct.pack("<I", dir_off)
    for pos, size, name in new_entries:
        out += struct.pack("<II8s", pos, size, name.encode("latin-1")[:8].ljust(8, b"\x00"))
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="base basic.wad (default: bundled ViZDoom scenarios)")
    ap.add_argument("--out", default="anydecision/games/data/arena300.wd")
    args = ap.parse_args()
    base = Path(args.base) if args.base else Path(vzd.scenarios_path) / "basic.wad"
    out = Path(args.out)
    if out.suffix != ".wad":
        out = out.with_suffix(".wad")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build(base)
    out.write_bytes(data)
    n_things = data.count(b"thing\n{")
    print(f"wrote {out} ({len(data)} bytes, {n_things} things: 1 player + 6 ClipBox + 3 Zombieman)")


if __name__ == "__main__":
    main()
