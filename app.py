==============================================================================
 BLOOD FOREST CRAWLER  -  Gesture-Controlled Pixel-Art RPG (1-Round Version)
==============================================================================
 A single-file, web-based pixel-art RPG played with real-time webcam hand
 gestures.  Designed to run inside Jupyter Notebook / Google Colab, where the
 browser owns the webcam + UI (HTML/CSS/JS) and Python owns the vision
 pipeline (OpenCV + MediaPipe) and the combat state machine.

 ARCHITECTURE
 ------------
   [Browser]  getUserMedia -> <video> -> <canvas>.toDataURL('image/jpeg')
        |                                        ^
        |  base64 JPEG (pulled by Python)        |  UI updates (pushed by
        v                                        |  Python as JS calls)
   [Python]   base64 -> np.frombuffer -> cv2.imdecode -> MediaPipe hand
                     landmarks -> gesture classification -> combat engine
                     -> JS bridge

 Hand tracking uses `mediapipe.tasks.vision.HandLandmarker` (works on
 mediapipe 1.x, where `mp.solutions` no longer exists) and falls back to the
 legacy `mp.solutions.hands` graph on older wheels.

 LAYOUT OF THIS FILE
 -------------------
   1.  Configuration & balance constants
   2.  Gesture recognition (MediaPipe / OpenCV)
   3.  Combat engine (pure Python, no I/O)
   4.  Front-end: CSS (theme, sprites, animations)
   5.  Front-end: HTML (stage, HUD, webcam, cooldown ring, combat log)
   6.  Front-end: JS (camera capture + UI mutation API)
   7.  JS bridge helpers (safe value escaping, eval_js abstraction)
   8.  Main game loop / entry point

 RUN
 ---
   Google Colab :  !pip install mediapipe opencv-python
                   %run blood_forest_crawler.py       (or paste into a cell)
   Jupyter      :  same, requires a browser with webcam permission
==============================================================================
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np

# MediaPipe is imported lazily-ish so the module can still be inspected without
# it, but the game itself requires it.
try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment guard
    mp = None

# Recent MediaPipe wheels ship only the Tasks API - `mp.solutions` may not
# exist at all - so the Tasks HandLandmarker is the primary backend and the
# legacy solution graph is used only when Tasks is unavailable.
HAND_LANDMARKER_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/main/"
    "mediapipe/tasks/testdata/vision/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "bfc", "hand_landmarker.task")


# =============================================================================
# 1.  CONFIGURATION & BALANCE
# =============================================================================

CAST_SECONDS = 3.0          # countdown a gesture must be held to cast
GESTURE_GRACE = 0.4         # tracking dropout tolerated before a cast breaks
COOLDOWN_SECONDS = 3.0      # recharge window after an action resolves
TARGET_FPS = 12             # frame pull rate from the browser
POTION_HEAL = (20, 30)      # inclusive heal range for POTION
STAGE_CLEAR_HEAL = 40       # HP restored when a monster dies

PLAYER_MAX_HP = 100

# Gesture identifiers, shared verbatim with the front-end.
SWORD, DAGGER, SHIELD, POTION = "SWORD", "DAGGER", "SHIELD", "POTION"
NONE = "NONE"

ACTION_ICONS = {SWORD: "\u2694\ufe0f", DAGGER: "\U0001f5e1\ufe0f",
                SHIELD: "\U0001f6e1\ufe0f", POTION: "\U0001f9ea"}

# Rock-paper-scissors table:  BEATS[a] is the action that `a` defeats.
BEATS = {SWORD: DAGGER, DAGGER: SHIELD, SHIELD: SWORD}


@dataclass
class Monster:
    """Static definition of one stage encounter."""
    name: str
    max_hp: int
    damage: int          # damage dealt to the player on a won/neutral exchange
    css_class: str       # front-end sprite selector
    color: str           # accent colour used for glow + HP bar theming
    face: str            # glyph shown inside the enemy portrait frame


# Single monster encounter for a 1-round game[cite: 1]
MONSTERS: List[Monster] = [
    Monster("BLOOD FIEND",     70,  8, "m-fiend",  "#ff3b30", "\U0001f479"),
]


# =============================================================================
# 2.  GESTURE RECOGNITION
# =============================================================================

# MediaPipe Hands landmark indices for finger tips and their PIP joints.
FINGER_TIPS = [8, 12, 16, 20]      # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
WRIST, MIDDLE_MCP = 0, 9


def ensure_hand_landmarker_model(path: str = HAND_LANDMARKER_PATH) -> str:
    """Download the HandLandmarker bundle once and cache it on disk."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    return path


class GestureRecognizer:
    """Detects hand landmarks and maps a hand pose to a combat action.

    Backend selection is automatic:
      * ``mediapipe.tasks.vision.HandLandmarker`` when available (current
        wheels, including builds that no longer ship ``mp.solutions``);
      * the legacy ``mp.solutions.hands.Hands`` graph otherwise.

    Both backends expose the same 21 landmark indices, so the pose logic below
    is shared.  Finger extension is measured *relative to hand scale*
    (wrist -> middle MCP distance) so classification is invariant to how close
    the player sits to the camera.
    """

    def __init__(self, detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.5,
                 model_path: Optional[str] = None) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is required: pip install mediapipe")

        self.backend: Optional[str] = None
        self._landmarker = None
        self._hands = None
        self._last_ts = -1          # Tasks VIDEO mode needs rising timestamps
        tasks_error: Optional[Exception] = None

        # --- preferred backend: Tasks HandLandmarker ---------------------
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model = ensure_hand_landmarker_model(
                model_path or HAND_LANDMARKER_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(
                options)
            self.backend = "tasks"
        except Exception as exc:            # noqa: BLE001 - fall back below
            tasks_error = exc

        # --- fallback backend: legacy solutions graph --------------------
        if self._landmarker is None:
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "hands"):
                raise RuntimeError(
                    "Could not initialise MediaPipe hand tracking: the Tasks "
                    f"backend failed ({tasks_error}) and this mediapipe build "
                    "exposes no `solutions.hands`. Try: "
                    "pip install -U 'mediapipe>=0.10.9'")
            self._hands = solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.backend = "solutions"

    # -- geometry helpers -------------------------------------------------
    @staticmethod
    def _dist(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _scale(self, lm) -> float:
        """Reference length for the hand; guards against divide-by-zero."""
        return max(self._dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)

    def _finger_open(self, lm, tip: int, pip: int) -> bool:
        """A finger counts as open when its tip is meaningfully farther from
        the wrist than its PIP joint, normalised by hand scale."""
        s = self._scale(lm)
        return (self._dist(lm[tip], lm[WRIST]) -
                self._dist(lm[pip], lm[WRIST])) / s > 0.28

    def _thumb_open(self, lm) -> bool:
        s = self._scale(lm)
        return (self._dist(lm[THUMB_TIP], lm[WRIST]) -
                self._dist(lm[THUMB_MCP], lm[WRIST])) / s > 0.42

    # -- backend dispatch --------------------------------------------------
    def _landmarks(self, rgb: np.ndarray, timestamp_ms: int):
        """Return the 21 landmarks of the first detected hand, or None."""
        if self._landmarker is not None:                    # Tasks API
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
            # The VIDEO running mode rejects non-increasing timestamps.
            timestamp_ms = max(timestamp_ms, self._last_ts + 1)
            self._last_ts = timestamp_ms
            try:
                result = self._landmarker.detect_for_video(image, timestamp_ms)
            except Exception:       # noqa: BLE001 - a dropped frame is not fatal
                return None
            return result.hand_landmarks[0] if result.hand_landmarks else None

        result = self._hands.process(rgb)                   # legacy API
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    # -- public API -------------------------------------------------------
    def classify(self, bgr_frame: np.ndarray,
                 timestamp_ms: Optional[int] = None):
        """Return (action, frame).  Action is NONE when there is no clear pose.

        ``timestamp_ms`` must be monotonically increasing for the Tasks VIDEO
        running mode; it is derived from the wall clock when omitted.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)

        lm = self._landmarks(rgb, timestamp_ms)
        if lm is None:
            return NONE, bgr_frame
        return self.action_for(lm), bgr_frame

    def action_for(self, lm) -> str:
        """Map landmarks to an action purely by how many fingers are raised.

        raised fingers (index/middle/ring/pinky) -> action
            0  + thumb down : SWORD   (closed fist)
            0  + thumb up   : POTION  (thumbs up)
            2               : DAGGER  (peace sign)
            4               : SHIELD  (open palm)
        anything else is ambiguous and reported as NONE.
        """
        raised = sum(self._finger_open(lm, t, p)
                     for t, p in zip(FINGER_TIPS, FINGER_PIPS))
        thumb = self._thumb_open(lm)

        if raised == 0:
            return POTION if thumb else SWORD
        if raised == 2:
            return DAGGER
        if raised == 4:
            return SHIELD
        return NONE

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        if self._hands is not None:
            self._hands.close()


# =============================================================================
# 3.  COMBAT ENGINE
# =============================================================================

@dataclass
class CombatEvent:
    """One resolved exchange, ready to be rendered by the front-end."""
    text: str
    player_hit: bool = False
    monster_hit: bool = False
    stage_cleared: bool = False
    game_over: bool = False
    victory: bool = False
    player_damage: int = 0      # HP the monster took off the player
    player_heal: int = 0        # HP restored to the player
    monster_damage: int = 0     # HP the player took off the monster
    crit: bool = False          # high roll -> gold, oversized floater


@dataclass
class GameState:
    player_hp: int = PLAYER_MAX_HP
    stage: int = 0
    monster_hp: int = field(init=False)
    finished: bool = False
    won: bool = False

    def __post_init__(self) -> None:
        self.monster_hp = self.monster.max_hp

    @property
    def monster(self) -> Monster:
        return MONSTERS[min(self.stage, len(MONSTERS) - 1)]


class CombatEngine:
    """Pure game logic: takes a player action, returns a CombatEvent."""

    def __init__(self, state: Optional[GameState] = None) -> None:
        self.state = state or GameState()

    # -- helpers ----------------------------------------------------------
    def _clamp_player(self) -> None:
        self.state.player_hp = max(0, min(PLAYER_MAX_HP, self.state.player_hp))

    def _monster_choice(self) -> str:
        """Simple AI: uniform pick among the three combat stances."""
        return random.choice([SWORD, DAGGER, SHIELD])

    # -- main resolution --------------------------------------------------
    def resolve(self, player_action: str) -> CombatEvent:
        st = self.state
        mon = st.monster
        foe_action = self._monster_choice()

        if player_action == POTION:
            heal = random.randint(*POTION_HEAL)
            st.player_hp += heal
            self._clamp_player()
            st.player_hp = max(0, st.player_hp - mon.damage)
            ev = CombatEvent(
                f"\U0001f9ea POTION +{heal} HP \u2014 {mon.name} strikes for "
                f"{mon.damage}!",
                player_hit=True, player_heal=heal, player_damage=mon.damage,
            )
        elif BEATS[player_action] == foe_action:
            dmg = random.randint(14, 22)
            st.monster_hp = max(0, st.monster_hp - dmg)
            ev = CombatEvent(
                f"{ACTION_ICONS[player_action]} {player_action} beats "
                f"{foe_action} \u2014 {dmg} damage!",
                monster_hit=True, monster_damage=dmg, crit=dmg >= 20,
            )
        elif player_action == foe_action:
            chip = max(1, mon.damage // 3)
            st.player_hp = max(0, st.player_hp - chip)
            ev = CombatEvent(
                f"\u2694 CLASH! Both chose {player_action} \u2014 you take "
                f"{chip} chip damage.",
                player_hit=True, player_damage=chip,
            )
        else:                                   # monster counters the player
            st.player_hp = max(0, st.player_hp - mon.damage)
            ev = CombatEvent(
                f"\U0001f480 {foe_action} counters your {player_action} "
                f"\u2014 {mon.damage} damage taken!",
                player_hit=True, player_damage=mon.damage,
            )

        # -- post-exchange state transitions -------------------------------
        if st.monster_hp <= 0:
            if st.stage >= len(MONSTERS) - 1:
                st.finished, st.won = True, True
                ev.text = f"\U0001f3c6 {mon.name} FALLS! THE FOREST IS CLEANSED!"
                ev.stage_cleared = ev.victory = True
            else:
                st.stage += 1
                st.monster_hp = st.monster.max_hp
                st.player_hp += STAGE_CLEAR_HEAL
                self._clamp_player()
                ev.player_heal += STAGE_CLEAR_HEAL
                ev.text = (f"\u2728 {mon.name} SLAIN! +{STAGE_CLEAR_HEAL} HP "
                           f"\u2014 {st.monster.name} EMERGES!")
                ev.stage_cleared = True
        elif st.player_hp <= 0:
            st.finished, st.won = True, False
            ev.text = f"\U0001f571 YOU FELL TO THE {mon.name}\u2026 GAME OVER"
            ev.game_over = True

        return ev


# =============================================================================
# 4.  FRONT-END: CSS  (Hextech/MOBA theme, sprites, animations)
# =============================================================================

GAME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Cinzel:wght@600;900&family=Barlow+Condensed:wght@500;700&display=swap');

/* ---- design tokens: League-style hextech palette -------------------- */
#bfc-root {
  --gold:       #c8aa6e;   --gold-lt: #f0e6d2;   --gold-dk: #785a28;
  --blue-0:     #010a13;   --blue-1:  #0a1428;   --blue-2:  #0f2233;
  --teal:       #0ac8b9;   --hp:      #17c964;   --enemy:   #e84057;
  --mana:       #0397ab;
}

#bfc-root, #bfc-root * { box-sizing: border-box; margin: 0; padding: 0; }

#bfc-root {
  width: 1080px; margin: 0 auto; padding: 0 0 16px;
  font-family: 'Barlow Condensed', sans-serif; color: var(--gold-lt);
  background:
    radial-gradient(ellipse 80% 50% at 50% 0%, #10263a 0%, transparent 70%),
    linear-gradient(180deg, #04101c 0%, var(--blue-0) 100%);
  border: 2px solid var(--gold-dk);
  box-shadow: 0 0 0 1px #000, 0 0 60px rgba(10,200,185,.10),
              inset 0 0 90px rgba(0,0,0,.8);
}

/* ================= HEADER: hextech banner ========================= */
.bfc-header {
  position: relative; height: 78px; display: grid; align-items: center;
  grid-template-columns: 160px 1fr 160px; padding: 0 26px;
  background: linear-gradient(180deg, #0a1c2b 0%, #061019 100%);
  border-bottom: 2px solid var(--gold-dk);
}
.bfc-header::after {         /* thin gold hairline under the banner */
  content: ''; position: absolute; left: 0; right: 0; bottom: -5px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: .55;
}
.bfc-title {
  font-family: 'Cinzel', serif; font-weight: 900; font-size: 30px;
  letter-spacing: 7px; text-transform: uppercase;
  background: linear-gradient(180deg, #f7ecd2 0%, var(--gold) 45%, #8a6d34 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 0 14px rgba(200,170,110,.45));
}
.bfc-title .rune { color: var(--teal); -webkit-text-fill-color: var(--teal);
                   filter: drop-shadow(0 0 10px var(--teal)); font-size: 22px; }

.bfc-header .bfc-title { text-align: center; }

/* stage pips, right column of the banner */
.bfc-stagebox { text-align: right; }
.bfc-stagebox .lbl { font-size: 11px; letter-spacing: 3px; color: #7a8fa6; }
.bfc-pips { display: flex; gap: 7px; margin-top: 6px; justify-content: flex-end; }
.bfc-pip {
  width: 15px; height: 17px; background: #14283c;
  border: 1px solid #2b4a63;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
}
.bfc-pip.on   { background: linear-gradient(180deg, var(--gold-lt), var(--gold));
                box-shadow: 0 0 10px rgba(200,170,110,.8); }
.bfc-pip.done { background: linear-gradient(180deg, #1c7a5a, #0d3b2c); }

/* timer chip, left column */
.bfc-timer { text-align: left; }
.bfc-timer .lbl { font-size: 11px; letter-spacing: 3px; color: #7a8fa6; }
.bfc-timer .val { font-family: 'Cinzel', serif; font-size: 20px;
                  color: var(--gold-lt); }

/* ================= STAGE: deep misty blood forest ================== */
.bfc-stage {
  position: relative; height: 372px; overflow: hidden;
  border-top: 1px solid rgba(200,170,110,.25);
  border-bottom: 2px solid var(--gold-dk);
  background:
    radial-gradient(ellipse 55% 30% at 24% 76%, rgba(186,160,220,.26), transparent 72%),
    radial-gradient(ellipse 62% 26% at 76% 82%, rgba(150,132,205,.22), transparent 74%),
    radial-gradient(ellipse 30% 66% at 50% 2%,  rgba(255,206,196,.15), transparent 72%),
    radial-gradient(circle at 10% -8%,  #7c1f2e 0%, #4a1120 34%, transparent 62%),
    radial-gradient(circle at 38% -12%, #9a2739 0%, #571423 32%, transparent 60%),
    radial-gradient(circle at 70% -8%,  #7c1f2e 0%, #481020 34%, transparent 62%),
    radial-gradient(circle at 96% -14%, #8d2334 0%, #4d1220 30%, transparent 58%),
    linear-gradient(180deg, #2e0c16 0%, #280c1a 24%, #22102e 52%,
                            #2a1436 72%, #150812 100%);
}
/* layered trunk silhouettes: far (hazy) + near (solid) */
.bfc-trees {
  position: absolute; inset: 0 0 66px 0; pointer-events: none;
  background:
    repeating-linear-gradient(90deg, transparent 0 118px,
      rgba(46,16,30,.55) 118px 132px, transparent 132px 250px),
    repeating-linear-gradient(90deg, transparent 0 64px,
      rgba(16,5,11,.9) 64px 82px, transparent 82px 156px);
  -webkit-mask-image: linear-gradient(180deg, #000 0%, #000 56%, transparent 95%);
          mask-image: linear-gradient(180deg, #000 0%, #000 56%, transparent 95%);
}
/* volumetric moon shaft */
.bfc-shaft {
  position: absolute; left: 44%; top: -40px; width: 240px; height: 380px;
  background: linear-gradient(180deg, rgba(255,225,225,.20), transparent 78%);
  transform: rotate(9deg); filter: blur(10px); pointer-events: none;
}
/* drifting ground mist */
.bfc-mist {
  position: absolute; left: -50%; bottom: 44px; width: 200%; height: 150px;
  background:
    radial-gradient(ellipse 24% 62% at 18% 60%, rgba(206,188,236,.24), transparent 70%),
    radial-gradient(ellipse 20% 55% at 55% 48%, rgba(186,166,224,.20), transparent 70%),
    radial-gradient(ellipse 26% 58% at 84% 62%, rgba(196,178,230,.18), transparent 70%);
  animation: mistDrift 26s linear infinite; pointer-events: none;
}
@keyframes mistDrift { from { transform: translateX(0); }
                       to   { transform: translateX(24%); } }
/* floating embers */
.bfc-ember {
  position: absolute; bottom: 60px; width: 3px; height: 3px; border-radius: 50%;
  background: #ffb36b; box-shadow: 0 0 8px 2px rgba(255,140,60,.55);
  animation: emberRise linear infinite; pointer-events: none;
}
@keyframes emberRise {
  0%   { transform: translateY(0) translateX(0); opacity: 0; }
  15%  { opacity: .9; }
  100% { transform: translateY(-260px) translateX(26px); opacity: 0; }
}
/* cinematic vignette over the whole scene */
.bfc-vignette {
  position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(ellipse 74% 68% at 50% 46%, transparent 40%,
                              rgba(0,0,0,.72) 100%);
}

/* ---------------- floor: wooden / dirt walkway ---------------------- */
.bfc-floor {
  position: absolute; left: 0; right: 0; bottom: 0; height: 66px; z-index: 3;
  background:
    linear-gradient(180deg, rgba(255,220,170,.18) 0 4px, transparent 4px),
    repeating-linear-gradient(90deg, #d46c24 0 48px, #9c4a17 48px 52px),
    #d46c24;
  border-top: 4px solid #f0a05a;
  box-shadow: inset 0 -18px 24px rgba(50,16,4,.7),
              0 -2px 18px rgba(0,0,0,.6);
}
.bfc-floor::after {          /* dirt scatter along the plank tops */
  content: ''; position: absolute; inset: 4px 0 auto 0; height: 10px;
  background: repeating-linear-gradient(90deg,
    rgba(255,190,130,.16) 0 6px, transparent 6px 22px);
}

/* ================= HEALTH FRAMES (portrait + bars) ================== */
.bfc-frame {
  position: absolute; top: 16px; z-index: 6; display: flex; gap: 10px;
  align-items: center; padding: 8px 12px; width: 380px;
  background: linear-gradient(180deg, rgba(4,14,24,.92), rgba(2,8,14,.88));
  border: 1px solid var(--gold-dk);
  box-shadow: 0 0 0 1px rgba(0,0,0,.9), 0 6px 22px rgba(0,0,0,.6),
              inset 0 0 26px rgba(10,200,185,.06);
}
.bfc-frame.left  { left: 18px; }
.bfc-frame.right { right: 18px; flex-direction: row-reverse; }

/* hexagonal champion portrait */
.bfc-portrait {
  position: relative; width: 62px; height: 68px; flex: none;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
  background: linear-gradient(180deg, var(--gold-lt), var(--gold-dk));
  display: flex; align-items: center; justify-content: center;
}
.bfc-portrait .inner {
  width: 56px; height: 62px;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; background: radial-gradient(circle at 50% 35%, #1d3a52, #06131f);
}
.bfc-portrait.enemy .inner { background: radial-gradient(circle at 50% 35%, #4a1220, #1a0509); }
.bfc-lvl {
  position: absolute; bottom: -6px; left: 50%; transform: translateX(-50%);
  min-width: 24px; padding: 1px 5px; font-size: 12px; font-weight: 700;
  text-align: center; color: #0a1428; background: var(--gold);
  border: 1px solid #000; z-index: 2;
}

.bfc-frame .meta { flex: 1; min-width: 0; }
.bfc-frame.right .meta { text-align: right; }
.bfc-nameline {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 8px; margin-bottom: 5px;
}
.bfc-frame.right .bfc-nameline { flex-direction: row-reverse; }
.bfc-name {
  font-family: 'Cinzel', serif; font-size: 15px; letter-spacing: 2px;
  color: var(--gold-lt); text-shadow: 0 0 10px rgba(200,170,110,.45);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.bfc-frame.right .bfc-name { color: #ffbfc6;
  text-shadow: 0 0 10px rgba(232,64,87,.55); }
.bfc-hpnum { font-size: 13px; color: #9fb4c7; letter-spacing: 1px; }

/* segmented MOBA health bar with trailing "damage ghost" */
.bfc-hpbar {
  position: relative; height: 17px; background: #06121c;
  border: 1px solid #2b4a63; box-shadow: inset 0 0 8px #000; overflow: hidden;
}
.bfc-hpghost {                 /* white/red trail that lags behind a hit */
  position: absolute; inset: 0; width: 100%;
  background: rgba(255,255,255,.55); transition: width .55s ease .18s;
}
.bfc-hpfill {
  position: absolute; inset: 0; width: 100%; transition: width .2s linear;
  background: linear-gradient(180deg, #3ff08a 0%, var(--hp) 48%, #0b8f45 100%);
  box-shadow: 0 0 12px rgba(23,201,100,.55);
}
.bfc-hpfill.enemy {
  background: linear-gradient(180deg, #ff7c8c 0%, var(--enemy) 48%, #8f1023 100%);
  box-shadow: 0 0 12px rgba(232,64,87,.55);
}
.bfc-hpticks {                 /* per-25 HP segment dividers */
  position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(90deg,
    transparent 0 24px, rgba(0,0,0,.55) 24px 25px);
}
.bfc-hpbar .gloss {
  position: absolute; inset: 0 0 auto 0; height: 6px; pointer-events: none;
  background: linear-gradient(180deg, rgba(255,255,255,.28), transparent);
}
/* secondary resource bar: hero = cast charge, enemy = threat */
.bfc-manabar {
  margin-top: 4px; height: 6px; background: #06121c; border: 1px solid #1b2f3f;
  position: relative; overflow: hidden; opacity: .8;
}
.bfc-manafill {
  position: absolute; inset: 0; width: 100%;
  background: linear-gradient(180deg, #2ab6c9, #06697a);
  transition: width .18s linear;
}
.bfc-manafill.threat { background: linear-gradient(180deg, #b45a2a, #6b2f12); }

/* ================= SPRITES ========================================= */
.bfc-actor { position: absolute; bottom: 58px; z-index: 4; }
.bfc-hero    { left: 150px; animation: pixelIdle 1.5s steps(2) infinite; }
.bfc-monster { right: 150px; animation: pixelIdle 1.9s steps(2) infinite; }
.bfc-hero.attacking { animation: pixelLunge .42s steps(3) 1; }
.bfc-monster.hit    { animation: pixelFlicker .45s steps(1) 3; }

@keyframes pixelIdle   { 0%,100% { transform: translateY(0); }
                         50%     { transform: translateY(-6px); } }
@keyframes pixelLunge  { 0%   { transform: translate(0,0); }
                         40%  { transform: translate(64px,-10px); }
                         70%  { transform: translate(30px,0); }
                         100% { transform: translate(0,0); } }
@keyframes pixelFlicker{ 0%,100% { filter: none; }
                         50% { filter: brightness(3) saturate(3)
                                       drop-shadow(0 0 16px #ff2d2d); } }

/* ground shadow blob under each actor */
.bfc-actor::after {
  content: ''; position: absolute; left: 50%; bottom: -12px; width: 92px;
  height: 16px; transform: translateX(-50%);
  background: radial-gradient(ellipse, rgba(0,0,0,.6), transparent 70%);
}

/* --- hero: plumed cream-armour knight with sword + kite shield --- */
.hero-sprite { position: relative; width: 104px; height: 124px;
               image-rendering: pixelated;
               filter: drop-shadow(0 6px 0 rgba(0,0,0,.35))
                       drop-shadow(0 0 16px rgba(120,200,255,.25)); }
.hero-sprite i { position: absolute; display: block; outline: 3px solid #241d12; }

/* helmet plume: three stacked green pixel tufts */
.h-plume  { left: 38px; top: -20px; width: 30px; height: 26px; outline: none;
            background: linear-gradient(180deg,#8ee07f,#3f8c3a);
            clip-path: polygon(50% 0,74% 22%,100% 58%,74% 100%,50% 74%,
                               26% 100%,0 58%,26% 22%);
            filter: drop-shadow(0 0 8px rgba(99,194,87,.6)); }
.h-helm   { left: 30px; top: 2px;  width: 46px; height: 32px;
            border-radius: 12px 12px 3px 3px;
            background: linear-gradient(180deg,#f7f1da 0 16px,#dcd1ab 16px 100%); }
/* dark cross-slit visor of the great helm */
.h-visor  { left: 37px; top: 12px; width: 32px; height: 18px; outline: none;
            background:
              linear-gradient(90deg, transparent 0 13px, #241d12 13px 19px,
                              transparent 19px 32px),
              linear-gradient(180deg, transparent 0 4px, #241d12 4px 10px,
                              transparent 10px 18px); }
.h-body   { left: 26px; top: 34px; width: 54px; height: 44px;
            background: linear-gradient(180deg,#efe7ca 0 10px,#d8ceaa 10px 100%); }
.h-pauld  { left: 18px; top: 34px; width: 70px; height: 13px;
            background: linear-gradient(180deg,#f6efd6,#cfc4a0); }
.h-belt   { left: 26px; top: 72px; width: 54px; height: 9px; background: #6d5a2c;
            outline: none; }
.h-legs   { left: 32px; top: 81px; width: 42px; height: 22px; background: #3b3626; }
.h-boots  { left: 26px; top: 99px; width: 54px; height: 14px; background: #221c10; }
/* raised sword: blade + crossguard + grip */
.h-sword  { left: 86px; top: 2px;  width: 12px; height: 50px; outline: none;
            background: linear-gradient(90deg,#ffffff 0 4px,#c3d1de 4px 8px,
                                        #7f8fa0 8px 12px);
            clip-path: polygon(50% 0,100% 16%,100% 100%,0 100%,0 16%);
            filter: drop-shadow(0 0 2px #241d12)
                    drop-shadow(0 0 10px rgba(210,240,255,.6)); }
.h-guard  { left: 76px; top: 52px; width: 32px; height: 8px; background: #c9a13f; }
.h-grip   { left: 87px; top: 60px; width: 10px; height: 18px; background: #7a5220; }
/* kite shield with pale blue face and steel rim */
.h-shield { left: -8px; top: 34px; width: 40px; height: 52px; outline: none;
            background: linear-gradient(180deg,#cfe9f5 0 58%,#84b6cd 58% 100%);
            border: 4px solid #4d7f96; border-radius: 10px 10px 18px 18px;
            box-shadow: inset 0 0 0 3px #eef8fd, 0 0 0 3px #241d12,
                        0 0 14px rgba(150,220,255,.45); }

/* --- monsters: hooded wraith with a glowing staff --- */
.mon-sprite { width: 138px; height: 150px; position: relative; }
/* hood + robe silhouette */
.mon-sprite .robe {
  position: absolute; left: 26px; top: 0; width: 92px; height: 150px;
  background: linear-gradient(180deg,#181125 0%,#0b0713 68%,#050308 100%);
  clip-path: polygon(38% 0,62% 0,80% 8%,90% 26%,88% 52%,96% 74%,92% 100%,
                     58% 100%,58% 82%,42% 82%,42% 100%,
                     8% 100%,4% 74%,12% 52%,10% 26%,20% 8%);
  box-shadow: inset 0 0 26px rgba(0,0,0,.9);
}
/* faint rim light along the hood */
.mon-sprite .robe::after {
  content: ''; position: absolute; inset: 0; opacity: .16;
  background: radial-gradient(ellipse 60% 30% at 50% 12%, currentColor, transparent 70%);
}
.mon-sprite .eyes {
  position: absolute; top: 50px; left: 44px; width: 56px; height: 15px;
  animation: eyeGlow 2.2s ease-in-out infinite;
}
.mon-sprite .eyes::before, .mon-sprite .eyes::after {
  content: ''; position: absolute; top: 0; width: 23px; height: 15px;
  background: currentColor; box-shadow: 0 0 18px currentColor;
}
.mon-sprite .eyes::before { left: 0;
  clip-path: polygon(0 30%,100% 0,100% 72%,0 100%); }
.mon-sprite .eyes::after  { right: 0;
  clip-path: polygon(0 0,100% 30%,100% 100%,0 72%); }
@keyframes eyeGlow { 0%,100% { opacity: .78; } 50% { opacity: 1; } }
/* forked staff carried in the left claw */
.mon-sprite .staff {
  position: absolute; left: 12px; top: 30px; width: 10px; height: 118px;
  background: linear-gradient(90deg,#a5732f 0 4px,#7a5220 4px 10px);
}
/* the two prongs cradling the orb */
.mon-sprite .staff::before, .mon-sprite .staff::after {
  content: ''; position: absolute; top: -14px; width: 7px; height: 22px;
  background: #8a6028;
}
.mon-sprite .staff::before { left: -11px; transform: rotate(18deg); }
.mon-sprite .staff::after  { left: 14px;  transform: rotate(-18deg); }
.mon-sprite .orb {
  position: absolute; left: -15px; top: -34px; width: 40px; height: 40px;
  border-radius: 50%;
  background: radial-gradient(circle at 38% 34%, #f0dcff 0 18%,
                              currentColor 45%, #3c116e 100%);
  box-shadow: 0 0 26px currentColor, 0 0 60px currentColor;
  animation: orbPulse 2.4s ease-in-out infinite;
}
@keyframes orbPulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.1); } }
/* per-stage aura colour (drives the eyes + orb via currentColor) */
.m-fiend   { color: #ff4d5e; filter: drop-shadow(0 0 20px rgba(255,77,94,.55)); }
.m-stalker { color: #a06cff; filter: drop-shadow(0 0 20px rgba(160,108,255,.55)); }
.m-dragon  { color: #35e0f5; filter: drop-shadow(0 0 22px rgba(53,224,245,.55)); }

/* floating combat text over the sprites */
.bfc-float {
  position: absolute; z-index: 8; pointer-events: none; font-weight: 700;
  font-size: 30px; font-family: 'Cinzel', serif;
  animation: floatUp 1.15s ease-out forwards;
  text-shadow: 0 2px 0 #000, 0 0 16px currentColor;
}
.bfc-float.dmg  { color: #ff5a6a; }
.bfc-float.heal { color: #4ef58f; }
.bfc-float.crit { color: #ffd76a; font-size: 38px; }
@keyframes floatUp {
  0%   { transform: translate(-50%,0) scale(.6); opacity: 0; }
  20%  { transform: translate(-50%,-14px) scale(1.15); opacity: 1; }
  100% { transform: translate(-50%,-74px) scale(1); opacity: 0; }
}

/* full-screen hit flash + stage banner */
.bfc-hitflash {
  position: absolute; inset: 0; z-index: 7; pointer-events: none; opacity: 0;
  background: radial-gradient(ellipse at 50% 50%, transparent 45%,
                              rgba(230,20,50,.55) 100%);
}
.bfc-hitflash.on { animation: hitFlash .42s ease-out 1; }
@keyframes hitFlash { 0% { opacity: 0; } 25% { opacity: 1; } 100% { opacity: 0; } }

.bfc-banner {
  position: absolute; left: 0; right: 0; top: 44%; z-index: 9; text-align: center;
  font-family: 'Cinzel', serif; font-size: 40px; letter-spacing: 10px;
  text-transform: uppercase; color: var(--gold-lt); opacity: 0;
  text-shadow: 0 0 26px rgba(200,170,110,.9), 0 4px 0 #000;
  pointer-events: none;
}
.bfc-banner.show { animation: bannerIn 2.1s ease-out 1; }
@keyframes bannerIn {
  0%   { opacity: 0; transform: scale(1.5); letter-spacing: 30px; }
  22%  { opacity: 1; transform: scale(1);   letter-spacing: 10px; }
  74%  { opacity: 1; }
  100% { opacity: 0; transform: scale(.96); }
}

/* ================= BOTTOM HUD ====================================== */
.bfc-hud { display: flex; gap: 14px; padding: 14px 18px 0; align-items: stretch; }

.bfc-card {
  position: relative; background: linear-gradient(180deg, #081826, #04101a);
  border: 1px solid var(--gold-dk);
  box-shadow: inset 0 0 30px rgba(0,0,0,.7), 0 0 0 1px #000;
}
/* hextech corner ticks on every card */
.bfc-card::before, .bfc-card::after {
  content: ''; position: absolute; width: 12px; height: 12px;
  border: 1px solid var(--gold); opacity: .8;
}
.bfc-card::before { left: -1px; top: -1px; border-right: 0; border-bottom: 0; }
.bfc-card::after  { right: -1px; bottom: -1px; border-left: 0; border-top: 0; }

/* --- webcam --- */
.bfc-cam { width: 310px; height: 238px; overflow: hidden; }
.bfc-cam video { width: 100%; height: 100%; object-fit: cover;
                 transform: scaleX(-1); display: block; filter: contrast(1.05); }
.bfc-cam .cap {
  position: absolute; left: 0; right: 0; top: 0; z-index: 2; padding: 4px 8px;
  font-size: 11px; letter-spacing: 3px; color: #9fe8ff;
  background: linear-gradient(180deg, rgba(2,10,18,.9), transparent);
}
.bfc-cam .rec { color: #ff5a6a; animation: recBlink 1.2s steps(2) infinite; }
@keyframes recBlink { 0%,100% { opacity: 1; } 50% { opacity: .15; } }
.bfc-cam .scan {          /* subtle scanline overlay for the feed */
  position: absolute; inset: 0; z-index: 1; pointer-events: none;
  background: repeating-linear-gradient(180deg,
    rgba(0,0,0,.18) 0 1px, transparent 1px 3px);
}

/* --- centre: cast ring + ability bar --- */
.bfc-center { flex: 1; display: flex; align-items: center; gap: 16px;
              padding: 12px 16px; }

/* --- right rail: counter wheel + run stats --- */
.bfc-rail { width: 186px; flex: none; padding: 12px 14px; }
.bfc-rail h4 {
  font-family: 'Cinzel', serif; font-size: 13px; letter-spacing: 3px;
  color: var(--gold); margin-bottom: 8px; font-weight: 600;
}
.bfc-rail .row {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 14px; letter-spacing: 1px; color: #a9c0d2;
  padding: 3px 0; border-bottom: 1px solid rgba(200,170,110,.10);
}
.bfc-rail .row b { color: var(--gold-lt); font-weight: 700; }
.bfc-rail .wheel {
  margin-top: 10px; font-size: 13px; line-height: 1.9; color: #9fb4c7;
  letter-spacing: 1px;
}
.bfc-rail .wheel span { color: var(--gold-lt); }
.bfc-rail .gest { margin-bottom: 12px; }
.bfc-rail .gest > div {
  padding: 5px 0; border-bottom: 1px dashed rgba(200,170,110,.14);
}
.bfc-rail .gest b {
  display: block; font-size: 10px; letter-spacing: 1px; color: #d8e6f2;
}
.bfc-rail .gest span {
  display: block; font-size: 9px; letter-spacing: .5px; color: var(--gold-lt);
  opacity: .85; margin-top: 2px;
}

.bfc-ring { position: relative; width: 150px; height: 150px; flex: none; }
.bfc-ring svg { transform: rotate(-90deg); }
.bfc-ring .track { fill: none; stroke: #0b1d2c; stroke-width: 9; }
.bfc-ring .rim   { fill: none; stroke: rgba(200,170,110,.35); stroke-width: 1; }
.bfc-ring .prog  { fill: none; stroke: var(--teal); stroke-width: 9;
                   stroke-linecap: round; transition: stroke-dashoffset .1s linear;
                   filter: drop-shadow(0 0 8px currentColor); }
.bfc-ring .inner {
  position: absolute; inset: 0; display: flex; flex-direction: column; gap: 6px;
  align-items: center; justify-content: center; text-align: center;
}
.bfc-ring .icon  { font-size: 30px; filter: drop-shadow(0 0 10px rgba(0,0,0,.8)); }
.bfc-ring .state { font-size: 14px; letter-spacing: 2px; color: var(--gold-lt);
                   white-space: pre-line; line-height: 1.25; }
.bfc-ring .sub   { font-size: 11px; letter-spacing: 2px; color: #6f889d; }

/* ability slots (Q/W/E/R style) */
.bfc-abilities { display: flex; gap: 10px; flex-wrap: nowrap; }
.bfc-ability {
  position: relative; width: 80px; height: 96px; flex: none;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
  background: linear-gradient(180deg, #2a2415, #0b1420);
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 2px; transition: transform .12s ease;
}
.bfc-ability .glyph { font-size: 28px; line-height: 1; margin-top: -6px;
                      filter: drop-shadow(0 2px 3px rgba(0,0,0,.8)); }
.bfc-ability .nm    { font-size: 12px; letter-spacing: 2px; color: #d5c398;
                      margin-top: 2px; }
.bfc-ability .key {                  /* the gesture that casts this slot */
  position: absolute; bottom: 11px; left: 0; right: 0; text-align: center;
  font-size: 8px; letter-spacing: 1px; color: var(--teal); opacity: .95;
}
.bfc-ability .veil {                 /* darkening + radial cooldown wipe */
  position: absolute; inset: 0; background: rgba(2,8,14,.78); opacity: 0;
  transition: opacity .15s linear;
}
.bfc-ability .fill {                 /* bottom-up hold-charge fill */
  position: absolute; left: 0; right: 0; bottom: 0; height: 0%;
  background: linear-gradient(180deg, rgba(10,200,185,.65), rgba(10,200,185,.2));
  transition: height .09s linear;
}
.bfc-ability.ready  { box-shadow: 0 0 0 2px var(--gold), 0 0 18px rgba(200,170,110,.35); }
.bfc-ability.active { transform: translateY(-4px) scale(1.06);
                      box-shadow: 0 0 0 2px var(--teal), 0 0 26px rgba(10,200,185,.6); }
.bfc-ability.cool .veil { opacity: 1; }
.bfc-ability.cast { animation: castPulse .5s ease-out 1; }
@keyframes castPulse {
  0% { box-shadow: 0 0 0 2px #fff, 0 0 40px rgba(255,255,255,.9); }
  100% { box-shadow: 0 0 0 2px var(--gold), 0 0 0 rgba(0,0,0,0); }
}

/* ================= COMBAT LOG ====================================== */
.bfc-log {
  margin: 12px 18px 0; padding: 12px 16px; min-height: 52px;
  display: flex; align-items: center; justify-content: center; gap: 10px;
  font-size: 17px; letter-spacing: 1px; text-align: center; color: #ffeccc;
  background: linear-gradient(90deg, rgba(4,14,24,.2), rgba(8,24,38,.95),
                              rgba(4,14,24,.2));
  border-top: 1px solid var(--gold-dk); border-bottom: 1px solid var(--gold-dk);
  text-shadow: 0 2px 4px #000;
}
.bfc-log.flash { animation: logFlash .55s ease-out 1; }
@keyframes logFlash {
  0%   { background: rgba(200,170,110,.28); letter-spacing: 3px; }
  100% { background: linear-gradient(90deg, rgba(4,14,24,.2), rgba(8,24,38,.95),
                                     rgba(4,14,24,.2)); letter-spacing: 1px; }
}
"""


# =============================================================================
# 5.  FRONT-END: HTML
# =============================================================================

GAME_HTML = """
<div id="bfc-root">

  <!-- ======================= HEADER ======================= -->
  <div class="bfc-header">
    <div class="bfc-timer">
      <div class="lbl">RUN TIME</div>
      <div class="val" id="bfc-clock">00:00</div>
    </div>

    <div class="bfc-title"><span class="rune">&#10023;</span>
      Blood Forest Crawler <span class="rune">&#10023;</span></div>

    <div class="bfc-stagebox">
      <div class="lbl">STAGE</div>
      <div class="bfc-pips" id="bfc-pips">
        <div class="bfc-pip on"></div>
      </div>
    </div>
  </div>

  <!-- ======================== STAGE ======================= -->
  <div class="bfc-stage" id="bfc-stagebox">
    <div class="bfc-shaft"></div>
    <div class="bfc-trees"></div>
    <div class="bfc-mist"></div>

    <!-- hero health frame -->
    <div class="bfc-frame left">
      <div class="bfc-portrait">
        <div class="inner">&#129497;</div>
        <div class="bfc-lvl" id="bfc-lvl">1</div>
      </div>
      <div class="meta">
        <div class="bfc-nameline">
          <span class="bfc-name">HERO</span>
          <span class="bfc-hpnum" id="bfc-php-txt">100 / 100</span>
        </div>
        <div class="bfc-hpbar">
          <div class="bfc-hpghost" id="bfc-php-ghost"></div>
          <div class="bfc-hpfill"  id="bfc-php"></div>
          <div class="bfc-hpticks"></div><div class="gloss"></div>
        </div>
        <div class="bfc-manabar"><div class="bfc-manafill" id="bfc-mana"></div></div>
      </div>
    </div>

    <!-- monster health frame -->
    <div class="bfc-frame right">
      <div class="bfc-portrait enemy">
        <div class="inner" id="bfc-mon-face">&#128127;</div>
        <div class="bfc-lvl" id="bfc-mlvl">1</div>
      </div>
      <div class="meta">
        <div class="bfc-nameline">
          <span class="bfc-name" id="bfc-mname">BLOOD FIEND</span>
          <span class="bfc-hpnum" id="bfc-mhp-txt">70 / 70</span>
        </div>
        <div class="bfc-hpbar">
          <div class="bfc-hpghost" id="bfc-mhp-ghost"></div>
          <div class="bfc-hpfill enemy" id="bfc-mhp"></div>
          <div class="bfc-hpticks"></div><div class="gloss"></div>
        </div>
        <div class="bfc-manabar"><div class="bfc-manafill threat" id="bfc-mmana"></div></div>
      </div>
    </div>

    <!-- combatants -->
    <div id="bfc-hero" class="bfc-actor bfc-hero">
      <div class="hero-sprite">
        <i class="h-shield"></i><i class="h-pauld"></i><i class="h-body"></i>
        <i class="h-belt"></i><i class="h-legs"></i><i class="h-boots"></i>
        <i class="h-helm"></i><i class="h-visor"></i><i class="h-plume"></i>
        <i class="h-sword"></i><i class="h-guard"></i><i class="h-grip"></i>
      </div>
    </div>

    <div id="bfc-mon" class="bfc-actor bfc-monster">
      <div id="bfc-mon-sprite" class="mon-sprite m-fiend">
        <div class="staff"><div class="orb"></div></div>
        <div class="robe"></div><div class="eyes"></div>
      </div>
    </div>

    <div class="bfc-floor"></div>
    <div class="bfc-vignette"></div>
    <div class="bfc-hitflash" id="bfc-hitflash"></div>
    <div class="bfc-banner" id="bfc-banner"></div>
  </div>

  <!-- ===================== BOTTOM HUD ===================== -->
  <div class="bfc-hud">
    <div class="bfc-card bfc-cam">
      <div class="cap"><span class="rec">&#9679;</span> LIVE VISION FEED</div>
      <div class="scan"></div>
      <video id="bfc-video" autoplay playsinline muted></video>
    </div>

    <div class="bfc-card bfc-center">
      <!-- cast / cooldown ring -->
      <div class="bfc-ring">
        <svg width="150" height="150">
          <circle class="rim"   cx="75" cy="75" r="70"></circle>
          <circle class="track" cx="75" cy="75" r="61"></circle>
          <circle id="bfc-prog" class="prog" cx="75" cy="75" r="61"
                  stroke-dasharray="383.3" stroke-dashoffset="383.3"></circle>
        </svg>
        <div class="inner">
          <div id="bfc-ring-icon" class="icon">&#9876;</div>
          <div id="bfc-ring-state" class="state">READY</div>
          <div class="sub" id="bfc-ring-sub">HOLD 3.0s</div>
        </div>
      </div>

      <!-- gesture ability bar -->
      <div class="bfc-abilities" id="bfc-abilities">
        <div class="bfc-ability ready" data-act="SWORD">
          <div class="veil"></div><div class="fill"></div>
          <div class="glyph">&#9876;&#65039;</div><div class="nm">SWORD</div>
          <div class="key">&#9994; FIST</div>
        </div>
        <div class="bfc-ability ready" data-act="DAGGER">
          <div class="veil"></div><div class="fill"></div>
          <div class="glyph">&#128481;&#65039;</div><div class="nm">DAGGER</div>
          <div class="key">&#9996; PEACE</div>
        </div>
        <div class="bfc-ability ready" data-act="SHIELD">
          <div class="veil"></div><div class="fill"></div>
          <div class="glyph">&#128737;&#65039;</div><div class="nm">SHIELD</div>
          <div class="key">&#128400; PALM</div>
        </div>
        <div class="bfc-ability ready" data-act="POTION">
          <div class="veil"></div><div class="fill"></div>
          <div class="glyph">&#129514;</div><div class="nm">POTION</div>
          <div class="key">&#128077; THUMB</div>
        </div>
      </div>
    </div>

    <!-- run stats + counter wheel -->
    <div class="bfc-card bfc-rail">
      <h4>GESTURE GUIDE</h4>
      <div class="gest">
        <div><b>&#9994; CLOSED FIST</b><span>&#9876;&#65039; SWORD &mdash; attack</span></div>
        <div><b>&#9996; PEACE SIGN</b><span>&#128481;&#65039; DAGGER &mdash; attack</span></div>
        <div><b>&#128400; OPEN PALM</b><span>&#128737;&#65039; SHIELD &mdash; block</span></div>
        <div><b>&#128077; THUMBS UP</b><span>&#129514; POTION &mdash; heal 20-30</span></div>
      </div>
      <h4>RUN STATS</h4>
      <div class="row"><span>CASTS</span><b id="bfc-st-casts">0</b></div>
      <div class="row"><span>DAMAGE DEALT</span><b id="bfc-st-dealt">0</b></div>
      <div class="row"><span>DAMAGE TAKEN</span><b id="bfc-st-taken">0</b></div>
      <div class="row"><span>HP RESTORED</span><b id="bfc-st-healed">0</b></div>
      <div class="wheel">
        <div>&#9876;&#65039; <span>SWORD</span> &gt; &#128481;&#65039; DAGGER</div>
        <div>&#128481;&#65039; <span>DAGGER</span> &gt; &#128737;&#65039; SHIELD</div>
        <div>&#128737;&#65039; <span>SHIELD</span> &gt; &#9876;&#65039; SWORD</div>
      </div>
    </div>
  </div>

  <!-- ======================== LOG ========================= -->
  <div id="bfc-log" class="bfc-log">THE BLOOD FOREST STIRS&hellip; RAISE YOUR HAND TO THE CAMERA.</div>

  <canvas id="bfc-canvas" style="display:none"></canvas>
</div>
"""


# =============================================================================
# 6.  FRONT-END: JAVASCRIPT  (capture + UI mutation API)
# =============================================================================

GAME_JS = """
<script>
(function () {
  const RING_CIRCUMFERENCE = 383.3;   // 2 * PI * r, r = 61
  const el = (id) => document.getElementById(id);
  const state = { stream: null, ready: false, t0: Date.now() };

  /* ---------- ambience: embers + run clock -------------------------- */
  (function spawnEmbers() {
    const stage = el('bfc-stagebox');
    if (!stage) return;
    for (let i = 0; i < 22; i++) {
      const e = document.createElement('div');
      e.className = 'bfc-ember';
      e.style.left = (Math.random() * 100) + '%';
      e.style.animationDuration = (5 + Math.random() * 7) + 's';
      e.style.animationDelay = (-Math.random() * 10) + 's';
      e.style.opacity = 0.3 + Math.random() * 0.7;
      stage.appendChild(e);
    }
  })();

  setInterval(function () {
    const s = Math.floor((Date.now() - state.t0) / 1000);
    el('bfc-clock').textContent =
      String(Math.floor(s / 60)).padStart(2, '0') + ':' +
      String(s % 60).padStart(2, '0');
  }, 1000);

  /* ---------- camera bootstrap: <video> -> hidden <canvas> ---------- */
  window.bfcInitCamera = async function (w, h) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: w, height: h }, audio: false });
      const v = el('bfc-video');
      v.srcObject = stream;
      state.stream = stream;
      await v.play();
      const c = el('bfc-canvas');
      c.width = w; c.height = h;
      state.ready = true;
      return 'OK';
    } catch (e) {
      return 'ERR:' + e.message;
    }
  };

  /* ---------- frame delivery to Python ------------------------------ */
  window.bfcCapture = function () {
    if (!state.ready) return '';
    const v = el('bfc-video'), c = el('bfc-canvas');
    if (!v.videoWidth) return '';
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.6);   // base64 JPEG data URL
  };

  window.bfcStopCamera = function () {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop());
    state.ready = false;
    return 'OK';
  };

  /* ---------- health frames ----------------------------------------- */
  window.bfcSetHP = function (playerHp, playerMax, monsterHp, monsterMax, monsterName) {
    const pPct = Math.max(0, playerHp / playerMax * 100);
    const mPct = Math.max(0, monsterHp / monsterMax * 100);
    el('bfc-php').style.width = pPct + '%';
    el('bfc-mhp').style.width = mPct + '%';
    /* ghost bars lag behind, LoL-style damage trails */
    el('bfc-php-ghost').style.width = pPct + '%';
    el('bfc-mhp-ghost').style.width = mPct + '%';
    el('bfc-php-txt').textContent = playerHp + ' / ' + playerMax;
    el('bfc-mhp-txt').textContent = monsterHp + ' / ' + monsterMax;
    el('bfc-mname').textContent = monsterName;
  };

  /* stage pips, level badges, enemy portrait glyph */
  window.bfcSetStage = function (stageIndex, total, monsterFace) {
    const pips = el('bfc-pips').children;
    for (let i = 0; i < pips.length; i++) {
      pips[i].className = 'bfc-pip' +
        (i < stageIndex ? ' done' : (i === stageIndex ? ' on' : ''));
    }
    el('bfc-lvl').textContent = stageIndex + 1;
    el('bfc-mlvl').textContent = stageIndex + 1;
    el('bfc-mon-face').innerHTML = monsterFace;
    /* enemy "threat" bar scales with the stage */
    el('bfc-mmana').style.width = (40 + stageIndex * 30) + '%';
  };

  /* ---------- cast ring + ability bar -------------------------------- */
  window.bfcSetRing = function (progress, stateText, icon, color, sub) {
    const p = Math.max(0, Math.min(1, progress));
    const ring = el('bfc-prog');
    ring.style.strokeDashoffset = (RING_CIRCUMFERENCE * (1 - p)).toFixed(1);
    ring.style.stroke = color;
    el('bfc-ring-state').textContent = stateText;
    el('bfc-ring-icon').innerHTML = icon;
    el('bfc-ring-sub').textContent = sub || '';
  };

  /* activeAction: gesture currently held ('' when none)
     hold: 0..1 charge of that gesture   cooling: true while recharging */
  window.bfcSetAbilities = function (activeAction, hold, cooling) {
    const slots = el('bfc-abilities').children;
    for (let i = 0; i < slots.length; i++) {
      const s = slots[i], act = s.dataset.act;
      const isActive = !cooling && act === activeAction;
      s.classList.toggle('cool', !!cooling);
      s.classList.toggle('ready', !cooling);
      s.classList.toggle('active', isActive);
      s.querySelector('.fill').style.height =
        (isActive ? Math.round(hold * 100) : 0) + '%';
    }
  };

  /* ---------- browser-side 3s cooldown countdown ---------------------
     Python only fires this once per cast; the ring, the digit and the
     ability veils then animate on the browser's own clock so the
     countdown stays smooth between frame round-trips.                  */
  let cdFrame = null, castFrame = null;

  const stopCast = function () {
    if (castFrame) { cancelAnimationFrame(castFrame); castFrame = null; }
  };

  /* channel countdown: the gesture must stay up until it reaches 0.0 */
  window.bfcStartCast = function (action, seconds, icon) {
    const total = Math.max(0.1, seconds) * 1000;
    const start = performance.now();
    stopCast();
    const step = function (now) {
      const left = Math.max(0, total - (now - start));
      const done = 1 - left / total;
      window.bfcSetRing(done, 'CASTING\\n' + action, icon, '#0ac8b9',
                        (left / 1000).toFixed(1) + 's TO CAST');
      window.bfcSetAbilities(action, done, false);
      if (left > 0) { castFrame = requestAnimationFrame(step); return; }
      castFrame = null;
      window.bfcSetRing(1, 'RELEASE', icon, '#17c964', 'CASTING NOW');
    };
    castFrame = requestAnimationFrame(step);
  };

  window.bfcCancelCast = function () {
    stopCast();
    window.bfcSetRing(0, 'EQUIP ITEM', '&#10067;', '#5f7a90', 'SHOW A GESTURE');
    window.bfcSetAbilities('', 0, false);
  };

  window.bfcStartCooldown = function (seconds) {
    const total = Math.max(0.1, seconds) * 1000;
    const start = performance.now();
    stopCast();
    if (cdFrame) { cancelAnimationFrame(cdFrame); cdFrame = null; }
    window.bfcSetAbilities('', 0, true);

    const step = function (now) {
      const left = Math.max(0, total - (now - start));
      window.bfcSetRing(left / total, 'COOLDOWN',
                        (left / 1000).toFixed(1), '#c8aa6e', 'SECONDS LEFT');
      window.bfcSetCharge(1 - left / total);
      if (left > 0) { cdFrame = requestAnimationFrame(step); return; }
      cdFrame = null;
      window.bfcSetRing(0, 'READY', '&#9876;', '#17c964', 'HOLD 3.0s');
      window.bfcSetAbilities('', 0, false);
      window.bfcSetCharge(1);
    };
    cdFrame = requestAnimationFrame(step);
  };

  window.bfcCastFx = function (action) {
    const slots = el('bfc-abilities').children;
    for (let i = 0; i < slots.length; i++) {
      if (slots[i].dataset.act !== action) continue;
      slots[i].classList.remove('cast'); void slots[i].offsetWidth;
      slots[i].classList.add('cast');
      setTimeout(((n) => () => n.classList.remove('cast'))(slots[i]), 600);
    }
  };

  /* ---------- feedback: log, banner, floaters, animations ----------- */
  window.bfcLog = function (text) {
    const log = el('bfc-log');
    log.textContent = text;
    log.classList.remove('flash'); void log.offsetWidth; log.classList.add('flash');
  };

  window.bfcBanner = function (text) {
    const b = el('bfc-banner');
    b.textContent = text;
    b.classList.remove('show'); void b.offsetWidth; b.classList.add('show');
  };

  /* kind: 'dmg' | 'heal' | 'crit'   who: 'hero' | 'monster' */
  window.bfcFloat = function (who, text, kind) {
    const stage = el('bfc-stagebox');
    const f = document.createElement('div');
    f.className = 'bfc-float ' + (kind || 'dmg');
    f.textContent = text;
    f.style.left = (who === 'hero' ? '190px' : 'calc(100% - 215px)');
    f.style.top = '150px';
    stage.appendChild(f);
    setTimeout(() => f.remove(), 1300);
  };

  /* running scoreboard in the right rail */
  window.bfcSetStats = function (casts, dealt, taken, healed) {
    el('bfc-st-casts').textContent  = casts;
    el('bfc-st-dealt').textContent  = dealt;
    el('bfc-st-taken').textContent  = taken;
    el('bfc-st-healed').textContent = healed;
  };

  /* hero charge bar mirrors the cast/cooldown gauge (0..1) */
  window.bfcSetCharge = function (progress) {
    el('bfc-mana').style.width = Math.round(progress * 100) + '%';
  };

  window.bfcSetMonsterSprite = function (cssClass) {
    const s = el('bfc-mon-sprite');
    s.className = 'mon-sprite ' + cssClass;
    s.innerHTML = '<div class="staff"><div class="orb"></div></div>' +
                  '<div class="robe"></div><div class="eyes"></div>';
  };

  /* one-shot animation helpers (class stripped once keyframes finish) */
  window.bfcAnimate = function (who) {
    const node = who === 'hero' ? el('bfc-hero') : el('bfc-mon');
    const cls  = who === 'hero' ? 'attacking' : 'hit';
    node.classList.remove(cls); void node.offsetWidth; node.classList.add(cls);
    setTimeout(() => node.classList.remove(cls), 700);
    if (who !== 'hero') {
      const fx = el('bfc-hitflash');
      fx.classList.remove('on'); void fx.offsetWidth; fx.classList.add('on');
    }
  };
})();
</script>
"""


# =============================================================================
# 7.  JS BRIDGE
# =============================================================================

def js_arg(value) -> str:
    """Serialise a Python value into a safe JavaScript literal.

    Everything goes through ``json.dumps`` so quotes, backslashes, newlines and
    non-ASCII glyphs (emoji in combat text!) can never break the generated
    ``eval_js`` snippet.  ``ensure_ascii`` keeps the payload transport-safe.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps("" if value is None else str(value), ensure_ascii=True)


class JSBridge:
    """Thin abstraction over Colab / Jupyter JavaScript evaluation."""

    def __init__(self) -> None:
        self._eval: Optional[Callable[[str], object]] = None
        try:                                    # Google Colab
            from google.colab import output as colab_output
            self._eval = colab_output.eval_js
            self.kind = "colab"
        except ImportError:
            try:                                # Classic Jupyter fallback
                from IPython.display import Javascript, display

                def _jupyter_eval(expr: str):
                    display(Javascript(expr))
                    return None

                self._eval = _jupyter_eval
                self.kind = "jupyter"
            except ImportError:                 # pragma: no cover
                self.kind = "none"

    @property
    def available(self) -> bool:
        return self._eval is not None

    def call(self, fn: str, *args, ignore_result: bool = False):
        """Invoke ``window.<fn>(...)`` with safely escaped arguments."""
        if self._eval is None:
            return None
        expr = f"{fn}({', '.join(js_arg(a) for a in args)})"
        try:
            result = self._eval(expr)
            return None if ignore_result else result
        except Exception as exc:                # pragma: no cover - UI only
            print(f"[bridge] {fn} failed: {exc}")
            return None


def decode_frame(data_url: Optional[str]) -> Optional[np.ndarray]:
    """base64 data-URL -> BGR ndarray (None when the browser sent nothing)."""
    if not data_url or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        buf = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =============================================================================
# 8.  GAME LOOP / ENTRY POINT
# =============================================================================

class BloodForestCrawler:
    """Glues the browser UI, the vision pipeline and the combat engine."""

    def __init__(self, cam_width: int = 480, cam_height: int = 360) -> None:
        self.cam_size = (cam_width, cam_height)
        self.bridge = JSBridge()
        self.engine = CombatEngine()
        self.recognizer: Optional[GestureRecognizer] = None
        # gesture-hold tracking
        self._held_action = NONE
        self._hold_started = 0.0
        self._last_seen = 0.0       # last frame the held gesture was visible
        self._cooldown_until = 0.0
        # right-rail scoreboard
        self._casts = 0
        self._dealt = 0
        self._taken = 0
        self._healed = 0

    # -- rendering ---------------------------------------------------------
    def render_ui(self) -> None:
        """Inject CSS + HTML + JS into the notebook output cell."""
        from IPython.display import HTML, display
        display(HTML(f"<style>{GAME_CSS}</style>{GAME_HTML}{GAME_JS}"))

    def push_hp(self) -> None:
        st = self.engine.state
        self.bridge.call("bfcSetHP", st.player_hp, PLAYER_MAX_HP,
                         st.monster_hp, st.monster.max_hp, st.monster.name,
                         ignore_result=True)

    def push_ring(self, progress: float, text: str, icon: str,
                  color: str, sub: str = "") -> None:
        self.bridge.call("bfcSetRing", round(progress, 3), text, icon, color,
                         sub, ignore_result=True)

    def push_stage(self) -> None:
        st = self.engine.state
        self.bridge.call("bfcSetStage", st.stage, len(MONSTERS),
                         st.monster.face, ignore_result=True)

    # -- per-frame logic ---------------------------------------------------
    def _update_hud(self, action: str, now: float) -> None:
        """Drive the cast ring and the ability bar from the input state."""
        cooling = now < self._cooldown_until
        hold = 0.0

        # while recharging or channelling, the browser owns the ring: its
        # countdown animates on its own clock (see bfcStartCooldown/bfcStartCast)
        if cooling or self._held_action != NONE:
            return

        self.push_ring(0.0, "EQUIP ITEM", "\u2753", "#5f7a90",
                       "SHOW A GESTURE")
        self.bridge.call("bfcSetAbilities", "", 0.0, False, ignore_result=True)
        self.bridge.call("bfcSetCharge", 1.0, ignore_result=True)

    def _commit(self, action: str, now: float) -> None:
        """Resolve one exchange and animate the outcome."""
        event = self.engine.resolve(action)
        self._cooldown_until = now + COOLDOWN_SECONDS
        self._held_action, self._hold_started, self._last_seen = NONE, 0.0, 0.0

        self.bridge.call("bfcCastFx", action, ignore_result=True)
        # the browser animates the 3s countdown on its own clock, so the ring
        # keeps ticking smoothly between our (slower) frame round-trips
        self.bridge.call("bfcStartCooldown", COOLDOWN_SECONDS,
                         ignore_result=True)
        if action != POTION:
            self.bridge.call("bfcAnimate", "hero", ignore_result=True)
        if event.monster_hit:
            self.bridge.call("bfcAnimate", "monster", ignore_result=True)

        # floating combat text, LoL-style
        if event.monster_damage:
            self.bridge.call("bfcFloat", "monster", f"-{event.monster_damage}",
                             "crit" if event.crit else "dmg", ignore_result=True)
        if event.player_heal:
            self.bridge.call("bfcFloat", "hero", f"+{event.player_heal}",
                             "heal", ignore_result=True)
        if event.player_damage:
            self.bridge.call("bfcFloat", "hero", f"-{event.player_damage}",
                             "dmg", ignore_result=True)

        if event.stage_cleared and not event.victory:
            self.bridge.call("bfcSetMonsterSprite",
                             self.engine.state.monster.css_class,
                             ignore_result=True)
            self.push_stage()
            self.bridge.call("bfcBanner", "STAGE CLEARED", ignore_result=True)
        elif event.victory:
            self.bridge.call("bfcBanner", "VICTORY", ignore_result=True)
        elif event.game_over:
            self.bridge.call("bfcBanner", "DEFEAT", ignore_result=True)

        self._casts += 1
        self._dealt += event.monster_damage
        self._taken += event.player_damage
        self._healed += event.player_heal
        self.bridge.call("bfcSetStats", self._casts, self._dealt,
                         self._taken, self._healed, ignore_result=True)

        self.push_hp()
        self.bridge.call("bfcLog", event.text, ignore_result=True)

    # -- main loop ---------------------------------------------------------
    def run(self, max_seconds: float = 900.0) -> GameState:
        if not self.bridge.available:
            raise RuntimeError(
                "No JavaScript bridge: run this inside Colab or Jupyter.")

        self.render_ui()
        self.recognizer = GestureRecognizer()

        init = self.bridge.call("bfcInitCamera", *self.cam_size)
        if isinstance(init, str) and init.startswith("ERR:"):
            self.bridge.call("bfcLog", f"CAMERA ERROR: {init[4:]}",
                             ignore_result=True)
            return self.engine.state

        self.push_hp()
        self.push_stage()
        self.bridge.call("bfcSetMonsterSprite",
                         self.engine.state.monster.css_class, ignore_result=True)
        self.bridge.call("bfcBanner", "STAGE 1", ignore_result=True)
        self.bridge.call("bfcLog",
                         "STAGE 1 \u2014 BLOOD FIEND BLOCKS THE PATH!",
                         ignore_result=True)

        frame_budget = 1.0 / TARGET_FPS
        deadline = time.time() + max_seconds

        try:
            while not self.engine.state.finished and time.time() < deadline:
                tick = time.time()

                frame = decode_frame(self.bridge.call("bfcCapture"))
                action = NONE
                if frame is not None:
                    action, _ = self.recognizer.classify(frame)

                now = time.time()
                if now >= self._cooldown_until:
                    if action != NONE:
                        if action != self._held_action:
                            # new gesture locked in: start the cast countdown
                            self._held_action = action
                            self._hold_started = self._last_seen = now
                            self.bridge.call("bfcStartCast", action,
                                             CAST_SECONDS,
                                             ACTION_ICONS[action],
                                             ignore_result=True)
                        else:
                            self._last_seen = now
                            if now - self._hold_started >= CAST_SECONDS:
                                self._commit(action, now)
                                continue
                    elif (self._held_action != NONE and
                          now - self._last_seen > GESTURE_GRACE):
                        # gesture lost for too long: the channel breaks
                        self._held_action, self._hold_started = NONE, 0.0
                        self.bridge.call("bfcCancelCast", ignore_result=True)

                self._update_hud(action, now)

                elapsed = time.time() - tick
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)
        except KeyboardInterrupt:
            self.bridge.call("bfcLog", "RUN ABANDONED.", ignore_result=True)
        finally:
            self.bridge.call("bfcStopCamera", ignore_result=True)
            if self.recognizer:
                self.recognizer.close()

        st = self.engine.state
        if st.finished:
            self.push_ring(1.0, "VICTORY" if st.won else "DEFEATED",
                           "\U0001f3c6" if st.won else "\U0001f480",
                           "#c8aa6e" if st.won else "#e84057",
                           "RUN COMPLETE")
            self.bridge.call("bfcSetAbilities", "", 0.0, True,
                             ignore_result=True)
        return st


def play(cam_width: int = 480, cam_height: int = 360,
         max_seconds: float = 900.0) -> GameState:
    """Convenience entry point: ``play()`` in a notebook cell."""
    return BloodForestCrawler(cam_width, cam_height).run(max_seconds)


if __name__ == "__main__":
    play()
