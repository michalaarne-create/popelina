from __future__ import annotations

from pathlib import Path
from unittest import TestCase, mock

import auto_main


class AutoMainScreenOnlyTests(TestCase):
    def _base_state(self, *, qid: str, title: str, question: str) -> dict:
        state = {
            "url": f"http://127.0.0.1:8000/t/{qid[4:6].lstrip('0') or '1'}/1",
            "title": title,
            "qid": qid,
            "question": question,
            "nextVisible": True,
            "nextBbox": [247, 452, 357, 489],
            "hasSelect": False,
            "selectBbox": None,
            "hasTextbox": False,
            "textboxValue": "",
            "textboxBbox": None,
            "options": [],
            "selectOptions": [],
            "textBlocks": [],
            "pageText": "",
        }
        state["signature"] = auto_main._build_signature(state)
        return state

    def test_build_screen_decision_uses_screen_only_page_state_for_single(self) -> None:
        state = self._base_state(
            qid="type01_q01",
            title="1) Jednokrotna odpowiedź + Next - 1/5",
            question="Jakiego koloru jest trawa?",
        )
        state["options"] = [
            {"index": 0, "text": "Czerwony", "kind": "radio", "checked": False, "bbox": [247, 242, 1193, 285]},
            {"index": 1, "text": "Zielony", "kind": "radio", "checked": False, "bbox": [247, 293, 1193, 336]},
            {"index": 2, "text": "Niebieski", "kind": "radio", "checked": False, "bbox": [247, 344, 1193, 387]},
        ]

        with mock.patch.object(auto_main, "_run_region_grow_for_image", side_effect=AssertionError("region_grow should not run")):
            decision = auto_main._build_screen_decision(screenshot_path=Path("dummy.png"), page_state=state)

        self.assertEqual(decision["trace"]["screen_parser_source"], "page_state_screen_only")
        self.assertEqual([a["kind"] for a in decision["actions"]], ["screen_click", "wait", "screen_click"])
        self.assertTrue(all(a["kind"] != "dom_select_option" for a in decision["actions"]))

    def test_build_screen_decision_uses_screen_only_page_state_for_dropdown(self) -> None:
        state = self._base_state(
            qid="type05_q01",
            title="5) Dropdown z Next - 1/5",
            question="Wybierz kolor trawy:",
        )
        state["hasSelect"] = True
        state["selectBbox"] = [247, 242, 1193, 285]
        state["selectOptions"] = [
            {"index": 0, "text": "", "value": "", "selected": True},
            {"index": 1, "text": "Czerwony", "value": "Czerwony", "selected": False},
            {"index": 2, "text": "Zielony", "value": "Zielony", "selected": False},
            {"index": 3, "text": "Niebieski", "value": "Niebieski", "selected": False},
        ]
        state["signature"] = auto_main._build_signature(state)

        with mock.patch.object(auto_main, "_run_region_grow_for_image", side_effect=AssertionError("region_grow should not run")):
            decision = auto_main._build_screen_decision(screenshot_path=Path("dummy.png"), page_state=state)

        self.assertEqual(decision["trace"]["screen_parser_source"], "page_state_screen_only")
        self.assertTrue(decision["trace"].get("fallback_used"))
        self.assertIn("visible_dropdown_options_as_choice_targets", decision["trace"].get("fallback_reasons") or [])
        self.assertTrue(all(a["kind"] != "dom_select_option" for a in decision["actions"]))

    def test_build_screen_decision_uses_screen_only_page_state_for_text(self) -> None:
        state = self._base_state(
            qid="type11_q01",
            title="11) Wpisywanie klawiaturą z Next - 1/5",
            question="Wpisz imię: Ala",
        )
        state["hasTextbox"] = True
        state["textboxBbox"] = [247, 242, 1193, 285]
        state["signature"] = auto_main._build_signature(state)

        with mock.patch.object(auto_main, "_run_region_grow_for_image", side_effect=AssertionError("region_grow should not run")):
            decision = auto_main._build_screen_decision(screenshot_path=Path("dummy.png"), page_state=state)

        self.assertEqual(decision["trace"]["screen_parser_source"], "page_state_screen_only")
        self.assertIn("type_answer", [a.get("reason") for a in decision["actions"]])
        self.assertTrue(all(a["kind"] != "dom_select_option" for a in decision["actions"]))

    def test_build_screen_decision_uses_screen_only_page_state_for_dropdown_scroll_auto(self) -> None:
        state = self._base_state(
            qid="type08_q01",
            title="8) Dropdown wymagający scrolla bez Next - 1/5",
            question="Wybierz właściwą opcję (scroll, auto):",
        )
        state["nextVisible"] = False
        state["nextBbox"] = None
        state["hasSelect"] = True
        state["selectBbox"] = [247, 242, 1193, 380]
        state["selectOptions"] = [{"index": 0, "text": "", "value": "", "selected": True}] + [
            {
                "index": idx,
                "text": ("Poprawna" if idx == 19 else f"Opcja {idx}"),
                "value": ("Poprawna" if idx == 19 else f"Opcja {idx}"),
                "selected": False,
            }
            for idx in range(1, 26)
        ]
        state["signature"] = auto_main._build_signature(state)

        with mock.patch.object(auto_main, "_run_region_grow_for_image", side_effect=AssertionError("region_grow should not run")):
            decision = auto_main._build_screen_decision(screenshot_path=Path("dummy.png"), page_state=state)

        self.assertEqual(decision["resolved_answer"]["question_type"], "dropdown_scroll")
        self.assertEqual([a["kind"] for a in decision["actions"]], ["screen_click"])
        self.assertTrue(all(a["kind"] != "dom_select_option" for a in decision["actions"]))

    def test_build_screen_decision_uses_screen_only_page_state_for_text_auto(self) -> None:
        state = self._base_state(
            qid="type12_q01",
            title="12) Wpisywanie klawiaturą bez Next - 1/5",
            question="Wpisz 3",
        )
        state["nextVisible"] = False
        state["nextBbox"] = None
        state["hasTextbox"] = True
        state["textboxBbox"] = [247, 242, 1193, 285]
        state["signature"] = auto_main._build_signature(state)

        with mock.patch.object(auto_main, "_run_region_grow_for_image", side_effect=AssertionError("region_grow should not run")):
            decision = auto_main._build_screen_decision(screenshot_path=Path("dummy.png"), page_state=state)

        self.assertEqual(decision["resolved_answer"]["question_type"], "text")
        self.assertIn("submit_text_answer", [a.get("reason") for a in decision["actions"]])
        self.assertTrue(all(a["kind"] != "dom_select_option" for a in decision["actions"]))
