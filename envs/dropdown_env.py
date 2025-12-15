"""
Dropdown Environments

- DropdownEnv: Existing cv2-rendered env (kept as-is).
- DropdownBBoxEnv: New simulated, no-screen env with dynamic bounding box
  observation and discrete actions: click, scroll_down, scroll_up.

The new DropdownBBoxEnv is tailored for PPO training where the agent only sees
the dropdown bounding box and state, and can only:
  0 = click
  1 = scroll_down
  2 = scroll_up

Task: open dropdown (click), scroll within it down to the bottom, then click to
select. Rewards increase for selecting lower options, encouraging scrolling.
Penalties discourage invalid scrolling/clicks and inefficiency.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
import cv2
import time
from typing import Any, Dict, List, Tuple, Optional


class DropdownEnv(gym.Env):
    """
    OPTIMIZED: cv2 rendering zamiast PIL (10x szybsze!)
    """
    
    def __init__(self, questions_file='data/quiz_dropdowns.json'):
        super().__init__()
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
            self.questions = data['questions']
        
        # ✅ ZMNIEJSZONY rozmiar renderowania
        self.screen_w = 640
        self.screen_h = 480
        
        # ACTION: [click_x, click_y]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # OBSERVATION: screen + state
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'state': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        
        # State
        self.current_question_idx = None
        self.dropdown_open = False
        self.scroll_position = 0
        self.step_count = 0
        self.max_steps = 30
        
        # Rendering constants (skalowane do 640x480)
        self.option_height = 30  # Zmniejszone z 40
        self.max_visible_options = 4
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_question_idx = self.np_random.integers(0, len(self.questions))
        self.dropdown_open = False
        self.scroll_position = 0
        self.step_count = 0
        
        obs = self._get_obs()
        info = {
            'question': self.questions[self.current_question_idx]['question'],
            'dropdown_open': False,
            'scroll_position': 0
        }
        
        return obs, info
    
    def _get_current_question(self):
        return self.questions[self.current_question_idx]
    
    def _render_screen(self):
        """FAST rendering - bez tekstu!"""
        
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255
        
        question = self._get_current_question()
        
        # Scale coords
        scale_x = self.screen_w / 1920
        scale_y = self.screen_h / 1080
        
        dropdown_x = int(question['x'] * scale_x)
        dropdown_y = int(question['y'] * scale_y)
        dropdown_w = int(question['width'] * scale_x)
        dropdown_h = self.option_height
        
        # ❌ USUŃ cv2.putText dla pytania!
        # cv2.putText(...)  # ZAKOMENTUJ
        
        if not self.dropdown_open:
            # Zamknięty - tylko prostokąt
            cv2.rectangle(img, (dropdown_x, dropdown_y), 
                        (dropdown_x + dropdown_w, dropdown_y + dropdown_h),
                        (51, 51, 51), 2)
            cv2.rectangle(img, (dropdown_x, dropdown_y), 
                        (dropdown_x + dropdown_w, dropdown_y + dropdown_h),
                        (245, 245, 245), -1)
            
            # ❌ USUŃ tekst "Select answer"
            # cv2.putText(...)  # ZAKOMENTUJ
            
        else:
            # Otwarty
            options = question['options']
            total_options = len(options)
            
            max_scroll_steps = max(0, total_options - self.max_visible_options)
            current_scroll_step = int(self.scroll_position * max_scroll_steps)
            
            start_idx = current_scroll_step
            end_idx = min(start_idx + self.max_visible_options, total_options)
            visible_height = (end_idx - start_idx) * self.option_height
            
            # Tło
            cv2.rectangle(img, (dropdown_x, dropdown_y),
                        (dropdown_x + dropdown_w, dropdown_y + visible_height),
                        (255, 255, 255), -1)
            cv2.rectangle(img, (dropdown_x, dropdown_y),
                        (dropdown_x + dropdown_w, dropdown_y + visible_height),
                        (51, 51, 51), 2)
            
            # Opcje - TYLKO prostokąty, BEZ tekstu
            for i in range(start_idx, end_idx):
                local_i = i - start_idx
                option_y = dropdown_y + local_i * self.option_height
                
                # Alternate color
                color = (250, 250, 250) if local_i % 2 == 0 else (240, 240, 240)
                cv2.rectangle(img, (dropdown_x, option_y),
                            (dropdown_x + dropdown_w, option_y + self.option_height),
                            color, -1)
                
                # ❌ USUŃ tekst opcji
                # cv2.putText(...)  # ZAKOMENTUJ
                
                # Separator
                if local_i < (end_idx - start_idx - 1):
                    cv2.line(img, (dropdown_x, option_y + self.option_height),
                            (dropdown_x + dropdown_w, option_y + self.option_height),
                            (220, 220, 220), 1)
            
            # Scroll indicator
            if max_scroll_steps > 0:
                indicator_h = 15
                indicator_y = dropdown_y + visible_height - indicator_h - 5
                color = (34, 139, 34) if self.scroll_position < 1.0 else (34, 34, 178)
                
                cv2.rectangle(img, (dropdown_x + dropdown_w - 20, indicator_y),
                            (dropdown_x + dropdown_w - 5, indicator_y + indicator_h),
                            color, -1)
        
        return img
    
    def _get_obs(self):
        screen = self._render_screen()
        
        question = self._get_current_question()
        
        # Skaluj koordynaty
        scale_x = self.screen_w / 1920
        scale_y = self.screen_h / 1080
        
        norm_dropdown_x = (question['x'] * scale_x) / self.screen_w * 2 - 1
        norm_dropdown_y = (question['y'] * scale_y) / self.screen_h * 2 - 1
        norm_dropdown_w = (question['width'] * scale_x) / self.screen_w * 2
        
        total_options = len(question['options'])
        max_scroll = max(0, total_options - self.max_visible_options)
        
        state = np.array([
            1.0 if self.dropdown_open else 0.0,
            self.scroll_position,
            norm_dropdown_x,
            norm_dropdown_y,
            norm_dropdown_w,
            total_options / 10.0,
            1.0 if max_scroll > 0 else 0.0,
            self.step_count / self.max_steps
        ], dtype=np.float32)
        
        return {
            'screen': screen,
            'state': state
        }
    
    def step(self, action):
        """Action: [click_x, click_y]"""
        
        # Skaluj akcję do screen size
        click_x = (action[0] + 1) / 2 * self.screen_w
        click_y = (action[1] + 1) / 2 * self.screen_h
        
        question = self._get_current_question()
        reward = 0.0
        terminated = False
        
        # Skaluj dropdown coords
        scale_x = self.screen_w / 1920
        scale_y = self.screen_h / 1080
        
        dropdown_x = question['x'] * scale_x
        dropdown_y = question['y'] * scale_y
        dropdown_w = question['width'] * scale_x
        dropdown_h = self.option_height
        
        action_taken = "none"
        
        if not self.dropdown_open:
            # Klik na zamkniętym dropdown
            if (dropdown_x <= click_x <= dropdown_x + dropdown_w and
                dropdown_y <= click_y <= dropdown_y + dropdown_h):
                self.dropdown_open = True
                reward += 10.0
                action_taken = "opened_dropdown"
            else:
                reward -= 1.0
                action_taken = "missed_dropdown"
        
        else:
            # Dropdown otwarty
            options = question['options']
            total_options = len(options)
            max_scroll_steps = max(0, total_options - self.max_visible_options)
            
            current_scroll_step = int(self.scroll_position * max_scroll_steps) if max_scroll_steps > 0 else 0
            start_idx = current_scroll_step
            end_idx = min(start_idx + self.max_visible_options, total_options)
            visible_height = (end_idx - start_idx) * self.option_height
            
            if (dropdown_x <= click_x <= dropdown_x + dropdown_w and
                dropdown_y <= click_y <= dropdown_y + visible_height):
                
                relative_y = click_y - dropdown_y
                clicked_option_local_idx = int(relative_y / self.option_height)
                
                if 0 <= clicked_option_local_idx < (end_idx - start_idx):
                    # Kliknięto opcję
                    clicked_option_global = start_idx + clicked_option_local_idx
                    
                    base_reward = 10.0
                    position_bonus = clicked_option_global * 5.0
                    option_reward = base_reward + position_bonus
                    
                    reward += option_reward
                    terminated = True
                    action_taken = f"selected_option_{clicked_option_global}"
                
                else:
                    # Scroll
                    if max_scroll_steps > 0:
                        old_scroll = self.scroll_position
                        self.scroll_position = min(1.0, self.scroll_position + (1.0 / max_scroll_steps))
                        
                        if self.scroll_position > old_scroll:
                            reward += 1.0
                            action_taken = "scrolled_down"
                        else:
                            reward -= 0.5
                            action_taken = "scroll_at_bottom"
                    else:
                        reward -= 0.5
                        action_taken = "clicked_between_options"
            else:
                # Klik poza
                reward -= 1.0
                action_taken = "clicked_outside"
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        info = {
            'question': question['question'],
            'dropdown_open': self.dropdown_open,
            'scroll_position': self.scroll_position,
            'click_pos': [click_x, click_y],
            'action_taken': action_taken,
            'reward': reward
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        screen = self._render_screen()
        cv2.imshow('Dropdown Environment', screen)
        cv2.waitKey(1)
    
    def close(self):
        cv2.destroyAllWindows()


class DropdownBBoxEnv(gym.Env):
    """
    Simulated dropdown with discrete actions and bounding-box observation only.

    Observation (Box, shape=(10,)) in [-1, 1]:
    [
      norm_x, norm_y, norm_w, norm_h,         # dropdown bbox (normalized)
      is_open,                                # 1=open, 0=closed
      scroll_norm,                            # scroll index normalized [0..1]
      total_norm,                             # total options ~ [0..1]
      visible_norm,                           # visible options ~ [0..1]
      step_progress                           # step_count / max_steps
    ]

    Actions (Discrete):
      0 = click
      1 = scroll_down
      2 = scroll_up

    Goal: click to open -> scroll down within dropdown to the bottom -> click.
    Rewards increase with lower (later) options to encourage scrolling.
    """

    def __init__(self,
                 screen_w: int = 1920,
                 screen_h: int = 1080,
                 min_options: int = 5,
                 max_options: int = 25,
                 min_visible: int = 3,
                 max_visible: int = 6,
                 max_steps: int = 50):
        super().__init__()

        self.screen_w = screen_w
        self.screen_h = screen_h

        self.min_options = min_options
        self.max_options = max_options
        self.min_visible = min_visible
        self.max_visible = max_visible

        self.max_steps = max_steps

        # Discrete actions: click, scroll_down, scroll_up
        self.action_space = spaces.Discrete(3)

        # Observation: bbox + state (no images)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        # Episode state
        self.dropdown_bbox = None  # (x, y, w, h) in screen coords
        self.dropdown_open = False
        self.total_options = None
        self.visible_count = None
        self.scroll_index = 0  # index of first visible option [0 .. total-visible]
        self.step_count = 0

        # Reward parameters (scaled for PPO stability later)
        self.r_open = 3.0
        self.r_scroll_step_base = 0.4
        self.r_bottom_bonus = 1.0
        self.r_click_base = 2.0
        self.r_click_scale = 8.0  # more reward the lower the option

        self.p_invalid_scroll = -0.4
        self.p_scroll_edge = -0.2
        self.p_click_when_closed = -0.5
        self.p_click_mid_not_bottom = -1.0
        self.p_step = -0.02  # time penalty

    # --- Helpers
    def _sample_bbox(self):
        # Randomize a dropdown bbox somewhere on the screen
        w = self.np_random.uniform(0.15, 0.35) * self.screen_w
        h = self.np_random.uniform(0.04, 0.06) * self.screen_h  # closed height
        x = self.np_random.uniform(0.05, 0.6) * self.screen_w
        y = self.np_random.uniform(0.15, 0.7) * self.screen_h
        return float(x), float(y), float(w), float(h)

    def _norm(self, x, a, b):
        return float(np.clip((x - a) / (b - a) * 2 - 1, -1, 1))

    def _get_obs(self):
        x, y, w, h = self.dropdown_bbox

        norm_x = self._norm(x, 0, self.screen_w)
        norm_y = self._norm(y, 0, self.screen_h)
        norm_w = self._norm(w, 0, self.screen_w)
        norm_h = self._norm(h, 0, self.screen_h)

        max_scroll = max(0, self.total_options - self.visible_count)
        scroll_norm = 0.0 if max_scroll == 0 else np.clip(self.scroll_index / max_scroll, 0, 1)

        total_norm = np.clip((self.total_options - self.min_options) / (self.max_options - self.min_options + 1e-6), 0, 1)
        visible_norm = np.clip((self.visible_count - self.min_visible) / (self.max_visible - self.min_visible + 1e-6), 0, 1)
        step_progress = np.clip(self.step_count / max(1, self.max_steps), 0, 1)

        obs = np.array([
            norm_x, norm_y, norm_w, norm_h,
            1.0 if self.dropdown_open else 0.0,
            scroll_norm,
            total_norm,
            visible_norm,
            step_progress,
            1.0 if (max_scroll > 0 and self.scroll_index >= max_scroll) else 0.0,
        ], dtype=np.float32)
        return obs

    # --- Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dropdown_bbox = self._sample_bbox()
        self.dropdown_open = False
        self.total_options = int(self.np_random.integers(self.min_options, self.max_options + 1))
        self.visible_count = int(self.np_random.integers(self.min_visible, min(self.max_visible, self.total_options) + 1))
        self.scroll_index = 0
        self.step_count = 0

        obs = self._get_obs()
        info = {
            'bbox': self.dropdown_bbox,
            'total_options': self.total_options,
            'visible_count': self.visible_count,
            'dropdown_open': self.dropdown_open,
        }
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        reward = 0.0
        terminated = False
        info = {}

        max_scroll = max(0, self.total_options - self.visible_count)

        if action == 0:  # click
            if not self.dropdown_open:
                # Open dropdown
                self.dropdown_open = True
                reward += self.r_open
                info['event'] = 'open_dropdown'
            else:
                # Selecting an option; to encourage scrolling, we assume the agent selects
                # the lowest currently visible option (index scroll_index+visible_count-1).
                selected_idx = min(self.total_options - 1, self.scroll_index + self.visible_count - 1)
                frac = 0.0 if self.total_options <= 1 else selected_idx / (self.total_options - 1)

                # Bigger reward for lower options
                reward += self.r_click_base + self.r_click_scale * frac

                # Encourage reaching true bottom before click
                if max_scroll > 0 and self.scroll_index >= max_scroll:
                    reward += self.r_bottom_bonus
                    info['event'] = 'click_select_bottom_visible'
                else:
                    # Penalize premature click (not at bottom yet)
                    reward += self.p_click_mid_not_bottom
                    info['event'] = 'click_select_not_bottom'

                terminated = True  # Episode ends after selection

        elif action == 1:  # scroll_down
            if not self.dropdown_open:
                reward += self.p_invalid_scroll
                info['event'] = 'scroll_down_closed'
            else:
                if max_scroll == 0:
                    reward += self.p_scroll_edge
                    info['event'] = 'scroll_down_no_room'
                elif self.scroll_index < max_scroll:
                    prev = self.scroll_index
                    self.scroll_index += 1
                    # progress-shaped reward
                    progress = (self.scroll_index / max_scroll) if max_scroll > 0 else 1.0
                    reward += self.r_scroll_step_base + 0.6 * progress
                    info['event'] = 'scroll_down'
                    # bonus when reaching bottom for the first time
                    if prev < max_scroll and self.scroll_index >= max_scroll:
                        reward += self.r_bottom_bonus
                else:
                    reward += self.p_scroll_edge
                    info['event'] = 'scroll_down_at_bottom'

        elif action == 2:  # scroll_up
            if not self.dropdown_open:
                reward += self.p_invalid_scroll
                info['event'] = 'scroll_up_closed'
            else:
                if max_scroll == 0:
                    reward += self.p_scroll_edge
                    info['event'] = 'scroll_up_no_room'
                elif self.scroll_index > 0:
                    self.scroll_index -= 1
                    # slight penalty (we want to encourage going down overall)
                    reward += -0.05
                    info['event'] = 'scroll_up'
                else:
                    reward += self.p_scroll_edge
                    info['event'] = 'scroll_up_at_top'

        # Step bookkeeping
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        # Small time penalty and small noise for exploration stability
        reward += self.p_step
        reward += float(self.np_random.uniform(-0.01, 0.01))

        # Scale down rewards for more stable PPO value targets (similar to hover env)
        reward /= 10.0

        obs = self._get_obs()
        info.update({
            'dropdown_open': self.dropdown_open,
            'scroll_index': self.scroll_index,
            'max_scroll': max_scroll,
            'bbox': self.dropdown_bbox,
            'selected_possible_index': (self.scroll_index + self.visible_count - 1) if self.dropdown_open else None,
        })

        return obs, reward, terminated, truncated, info

    def render(self):
        # No rendering for the simulation variant (screenless)
        pass

    def close(self):
        pass


"""Unified synthetic dropdown generator (uses utils.dropdown_ui_shared)"""
from utils.dropdown_ui_shared import render_grid


def save_40_dropdowns_debug(out_dir: str | Path = "debug", *, seed: Optional[int] = None) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas, payload = render_grid(canvas_w=1000, canvas_h=700, cols=5, rows=8, seed=seed, expanded_prob=0.6, limit=40)
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = out_dir / f"dropdowns_{ts}.png"
    json_path = out_dir / f"dropdowns_meta_{ts}.json"
    cv2.imwrite(str(img_path), canvas)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return img_path, json_path


if __name__ == "__main__":
    try:
        img_p, json_p = save_40_dropdowns_debug()
        print(f"Saved: {img_p} and {json_p}")
    except Exception as e:
        print(f"Generator error: {e}")
