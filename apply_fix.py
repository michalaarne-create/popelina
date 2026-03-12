# -*- coding: utf-8 -*-
import os

filepath = r"e:\BOT ANK\bot\moje_AI\yolov8\FULL BOT_BACKUP_2026-02-05_000842\popelina_github\scripts\region_grow\numpy_rate\rating.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Define the old function body exactly as it likely is
old_start = '        best = None\n        best_score = 0.0\n\n        for candidate in (bin_inv, bin_nrm):'
# We'll search for this anchor

if old_start not in content:
    # Try with different line endings
    old_start = old_start.replace('\n', '\r\n')

if old_start not in content:
    print("Could not find anchor!")
    exit(1)

new_code = """        best_circle_score = 0.0
        best_square_score = 1.0  # Temporary to test if replace works

        for candidate in (bin_inv, bin_nrm):
            contours, _ = cv2.findContours(candidate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            roi_area = float(rh * rw)
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < 8.0 or area > roi_area * 0.85:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w < 4 or h < 4:
                    continue
                ratio = float(w) / max(1.0, float(h))
                if ratio < 0.3 or ratio > 3.0:
                    continue
                peri = float(cv2.arcLength(c, True))
                if peri <= 0.0:
                    continue
                circularity = float((12.56637 * area) / (peri * peri + 1e-6))
                extent = float(area / (w * h + 1e-6))
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                vertices = len(approx)

                # ── BIAS NA KSZTAŁT ──────────────────────────────────────────
                c_score = max(0.0, min(1.0, (circularity - 0.65) / 0.30))
                s_score = max(0.0, min(1.0, 1.1 - abs(ratio - 1.0) / 0.5))
                s_score = min(s_score, max(0.0, min(1.0, (extent - 0.40) / 0.50)))
                
                if extent > 0.81: c_score *= 0.4
                if vertices == 4:
                    s_score = max(s_score, 0.90)
                
                if hough_circle_found and circularity > 0.6:
                    h_boost = 0.25 + (hough_conf - 0.7) * 0.5
                    c_score = max(c_score, min(0.95, c_score + h_boost))

                best_circle_score = max(best_circle_score, c_score)
                best_square_score = max(best_square_score, s_score)

        # ── FINALNA DECYZJA ──────────────────────────────────────────────────
        max_total = max(best_circle_score, best_square_score)
        if max_total < 0.25 and not hough_circle_found:
            return out

        # Bias na kwadrat (częsta obecność kółek/ptaszków wewnątrz checkboxów)
        if best_square_score > 0.70 and best_square_score >= (best_circle_score - 0.20):
            out["marker_shape"] = "square"
            out["marker_kind"] = "checkbox"
            out["marker_conf"] = round(best_square_score, 4)
        elif best_circle_score > best_square_score:
            out["marker_shape"] = "circle"
            out["marker_kind"] = "radio"
            out["marker_conf"] = round(best_circle_score, 4)
        else:
            out["marker_shape"] = "square"
            out["marker_kind"] = "checkbox"
            out["marker_conf"] = round(best_square_score, 4)

        return out"""

# We'll replace everything from old_start to the return out
# Let's find end also
end_anchor = '        return out\n    except Exception:'
if end_anchor not in content:
   end_anchor = end_anchor.replace('\n', '\r\n')

if end_anchor not in content:
    print("Could not find end anchor!")
    exit(1)

part1 = content.split(old_start)[0]
part2 = content.split(end_anchor)[1]

final_content = part1 + new_code + "\n" + end_anchor + part2

with open(filepath, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Replacement successful!")
