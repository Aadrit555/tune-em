"""Build arena300.wad: basic.wad geometry + 300-round ammo stockpile.

Reproducible UDMF edit of ViZDoom's bundled basic.wad (PWAD, MAP01):
  - 6x ClipBox (DoomEd 2048, 50 bullets each = 300 rounds) ringed around the
    player start for genuine engine pickup (ITEMCOUNT-verified, never conjured).
  - 7 live targets (3x Zombieman, 2x ShotgunGuy, 2x Imp) at fixed bearings.
Only geometry is reused from basic.wad. The ACS BEHAVIOR lump is deliberately
DROPPED: its Exit_Normal-on-kill script would end the episode the moment the
first target dies, before the Zombiemen and the stockpile come into play.
Without it the episode runs the full step budget (or until player death),
kills come from KILLCOUNT, and reward is raw engine reward. THINGS edits
never invalidate the BSP, and ZDoom rebuilds nodes from TEXTMAP.

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
# Mixed sparring roster (DoomEd, HP): fast kills up close, tougher ones out.
ROSTER: list[tuple[int, float, float]] = [
    (3004, 170.0, 20.0),  # Zombieman x3, inner ring
    (3004, 170.0, 140.0),
    (3004, 170.0, 260.0),
    (9, 300.0, 80.0),  # ShotgunGuy x2, outer ring
    (9, 300.0, 200.0),
    (3001, 300.0, 320.0),  # Imp x2, outer ring
    (3001, 190.0, 10.0),
]
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
    # 6 ClipBoxes (300 rounds): 2 within spawn contact range so the opening
    # turns collect them by genuine walk-over, 4 on a 96-unit ring for
    # mid-game collection during combat movement.
    for dx, dy in ((20.0, 0.0), (-20.0, 0.0)):
        text += _thing(px + dx, py + dy, CLIPBOX)
    for k in range(4):
        ang = math.radians(90.0 * k + 45.0)
        text += _thing(px + 96.0 * math.cos(ang), py + 96.0 * math.sin(ang), CLIPBOX)
    # Mixed roster on inner/outer rings (fixed bearings, deterministic map).
    for type_id, radius, deg in ROSTER:
        ang = math.radians(deg)
        text += _thing(px + radius * math.cos(ang), py + radius * math.sin(ang), type_id, angle=0)
    lumps["TEXTMAP"] = text.encode("utf-8")

    # Emit MAP01 + TEXTMAP + ENDMAP only. BEHAVIOR/SCRIPTS/DIALOGUE/ZNODES
    # are dropped (no exit-script, nodes rebuilt by ZDoom from TEXTMAP).
    keep = ["MAP01", "TEXTMAP", "ENDMAP"]
    # Rebuild the WAD byte layout (lumps back-to-back, fresh directory).
    out = bytearray(b"PWAD" + struct.pack("<II", 0, 0))
    new_entries = []
    for _, _, name in entries:
        if name not in keep:
            continue
        data = lumps[name]
        new_entries.append((len(out), len(data), name))
        out += data
    out[4:8] = struct.pack("<I", len(new_entries))
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
    print(f"wrote {out} ({len(data)} bytes, {n_things} things: 1 player + 6 ClipBox + 7 monsters)")


if __name__ == "__main__":
    main()
