"""Pet sprite geometry + animation-state taxonomy.

These values are *constants of the petdex format*, not per-pet data — the
real ``pet.json`` only carries ``id``/``displayName``/``description``/
``spritesheetPath``.  The official petdex web app and desktop client both
hardcode 192×208 frames, 6 frames per state, a 1100ms loop, and a 0.7 render
scale; we match them so installed pets animate identically.
"""

from __future__ import annotations

from enum import Enum

# Frame geometry (pixels).  A standard petdex spritesheet is a 1536×1872 grid
# → 8 columns × 9 rows of these frames.
FRAME_W = 192
FRAME_H = 208

# Frames consumed per animation state (the petdex web app uses CSS
# ``steps(6)``).  A sheet may physically contain more columns; we only step
# through the first ``FRAMES_PER_STATE``.
FRAMES_PER_STATE = 6

# Full-loop duration for one state, milliseconds (petdex default).
LOOP_MS = 1100

# Default on-screen scale relative to native frame size.  ``display.pet.scale``
# is the single master scalar: the desktop canvas multiplies its native pixels
# by it and every terminal surface derives its half-block/kitty column width
# from it (see :func:`cols_for_scale`), so one number shrinks all three
# interfaces together.  (petdex's own clients render at 0.7; we default smaller
# so the kitty/GUI mascot stays a glanceable corner sprite.  The half-block
# fallback can't shrink as far — see ``UNICODE_MIN_COLS`` — and clamps to its
# legibility floor instead.)
DEFAULT_SCALE = 0.33

# Terminal cells one native frame spans at ``scale == 1.0``.  A cell is ~8px
# wide, a frame is ``FRAME_W`` (192) px → 24 cells.  This mirrors the kitty
# graphics placement (``scaled_px // 8``) so at full scale every renderer agrees.
BASE_UNICODE_COLS = FRAME_W // 8

# Legibility floor for the half-block fallback.  A half-block cell samples the
# sprite at only 1 horizontal + 2 vertical taps, so below this width a 192×208
# pet collapses into an unreadable blob *regardless* of scale.  kitty/GUI draw
# true pixels and have no such floor — that's why the same ``scale: 0.33`` is
# crisp there but mush in half-blocks.  ``scale`` shrinks the unicode pet down
# TO this floor (and grows it above), instead of past it into noise.
UNICODE_MIN_COLS = 16


def cols_for_scale(scale: float) -> int:
    """Half-block width implied by *scale*, clamped to the legibility floor.

    Above the floor it tracks the kitty cell box (``scaled_px // 8``) so the two
    renderers converge at larger sizes; below it the floor keeps the sprite
    readable rather than letting it devolve into a blob.
    """
    return max(UNICODE_MIN_COLS, round(BASE_UNICODE_COLS * (scale or DEFAULT_SCALE)))


def resolve_cols(scale: float, unicode_cols: int = 0) -> int:
    """Resolve terminal width: explicit *unicode_cols* override, else from *scale*."""
    return int(unicode_cols) if unicode_cols and int(unicode_cols) > 0 else cols_for_scale(scale)


class PetState(str, Enum):
    """Animation state a pet can be shown in.

    Values are the petdex spritesheet *row names*.  Membership maps directly
    onto :data:`STATE_ROWS` (row index = position in that list).
    """

    IDLE = "idle"
    WAVE = "wave"
    RUN = "run"
    FAILED = "failed"
    REVIEW = "review"
    JUMP = "jump"


# Row order in the spritesheet (top → bottom).  Index of a state name here is
# the pixel row it occupies: ``row_y = STATE_ROWS.index(state) * FRAME_H``.
# ``extra1``/``extra2`` are reserved petdex rows we don't drive yet but keep so
# row math stays correct for sheets that include them.
STATE_ROWS: list[str] = [
    PetState.IDLE.value,
    PetState.WAVE.value,
    PetState.RUN.value,
    PetState.FAILED.value,
    PetState.REVIEW.value,
    PetState.JUMP.value,
    "extra1",
    "extra2",
]


def state_row_index(state: "PetState | str") -> int:
    """Return the spritesheet row index for *state* (clamped, never raises)."""
    value = state.value if isinstance(state, PetState) else str(state)
    try:
        return STATE_ROWS.index(value)
    except ValueError:
        return 0  # fall back to the idle row
