import asyncio
import contextlib
import io
import json
import random
import time
import gc
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except Exception:
    Image = None
    ImageEnhance = None
    ImageFilter = None

    ImageOps = None

try:
    import pytesseract
except Exception:
    pytesseract = None

if __package__:
    from .ai_recorder_common import (
        OCR_INTERVAL,
        OcrLine,
        OCR_LANG,
        WATCHDOG_TIMEOUT,
        QuestionGroup,
        WindowInfo,
        TabInfo,
        build_bbox,
        md5,
        norm_text,
        extract_url_candidates,
        domain_from_url,
        fuzzy_ratio,
        log,
    )
else:
    from ai_recorder_common import (
        OCR_INTERVAL,
        OcrLine,
        OCR_LANG,
        WATCHDOG_TIMEOUT,
        QuestionGroup,
        WindowInfo,
        TabInfo,
        build_bbox,
        md5,
        norm_text,
        extract_url_candidates,
        domain_from_url,
        fuzzy_ratio,
        log,
    )


class LiveRecorderCaptureMixin:
    save_min_interval: float = 0.5

    async def _extract_dom_text(self, page=None) -> Tuple[str, List[Dict[str, Any]]]:
        """Wydobywa tekst z DOM pomijając radio/checkbox."""
        js = r"""
            const elements = [];
            const texts = [];
            const processedTexts = new Set();
            
            const isElementReallyVisible = (el) => {
                if (!el) return false;
                
                const style = getComputedStyle(el);
                
                if (style.display === 'none' || 
                    style.visibility === 'hidden' ||
                    parseFloat(style.opacity) < 0.01) {
                    return false;
                }
                
                const rect = el.getBoundingClientRect();
                
                if (rect.width <= 0 || rect.height <= 0) {
                    return false;
                }
                
                if (rect.bottom < 0 || rect.top > window.innerHeight ||
                    rect.right < 0 || rect.left > window.innerWidth) {
                    return false;
                }
                
                const points = [
                    { x: rect.left + rect.width * 0.5, y: rect.top + rect.height * 0.5 },
                    { x: rect.left + 10, y: rect.top + 10 },
                    { x: rect.right - 10, y: rect.top + 10 },
                    { x: rect.left + rect.width * 0.5, y: rect.top + 5 }
                ];
                
                let visiblePoints = 0;
                
                for (const point of points) {
                    if (point.x < 0 || point.y < 0 || 
                        point.x > window.innerWidth || point.y > window.innerHeight) {
                        continue;
                    }
                    
                    const topEl = document.elementFromPoint(point.x, point.y);
                    
                    if (!topEl) continue;
                    
                    if (topEl === el || el.contains(topEl) || topEl.contains(el)) {
                        visiblePoints++;
                        continue;
                    }
                    
                    const topClass = (topEl.className || '').toString().toLowerCase();
                    const topId = (topEl.id || '').toLowerCase();
                    
                    if (topClass.includes('overlay') || 
                        topClass.includes('backdrop') || 
                        topClass.includes('modal-backdrop') ||
                        topId.includes('overlay') ||
                        topId.includes('backdrop')) {
                        return false;
                    }
                }
                
                return visiblePoints > 0;
            };
            
            const isHoneypot = (el) => {
                if (!el) return false;
                
                const inlineStyle = el.getAttribute('style') || '';
                if (inlineStyle.includes('opacity: 1e-') || 
                    inlineStyle.includes('opacity: 0.0000')) {
                    return true;
                }
                
                const className = (el.className || '').toString();
                if (className.includes('honeypot') ||
                    (className.includes('absolute') && 
                    className.includes('select-none') && 
                    className.includes('text-[5px]'))) {
                    return true;
                }
                
                const style = getComputedStyle(el);
                const fontSize = parseFloat(style.fontSize);
                if (fontSize < 6) {
                    return true;
                }
                
                return false;
            };
            
            const isClickable = (el) => {
                if (!el) return false;
                
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                
                if (['button', 'a', 'input', 'select', 'textarea'].includes(tag)) {
                    return true;
                }
                
                if (['button', 'link', 'checkbox', 'radio', 'option'].includes(role)) {
                    return true;
                }
                
                if (el.onclick || el.getAttribute('onclick')) {
                    return true;
                }
                
                const style = getComputedStyle(el);
                if (style.cursor === 'pointer') {
                    return true;
                }
                
                return false;
            };
            
            const isRadioOrCheckboxText = (el) => {
                if (!el) return false;
                
                const className = (el.className || '').toString().toLowerCase();
                if (className.includes('radio-text') || 
                    className.includes('checkbox-text') ||
                    className.includes('p-radio-text') ||
                    className.includes('p-checkbox-text') ||
                    className.includes('form-check-label')) {
                    return true;
                }
                
                const testId = (el.getAttribute('data-test-id') || '').toLowerCase();
                if (testId.includes('radio') || 
                    testId.includes('checkbox') ||
                    testId.includes('single_choice') ||
                    testId.includes('multiple_choice')) {
                    return true;
                }
                
                if (el.tagName && el.tagName.toLowerCase() === 'label') {
                    const forAttr = el.getAttribute('for');
                    if (forAttr) {
                        const input = document.getElementById(forAttr);
                        if (input && (input.type === 'radio' || input.type === 'checkbox')) {
                            return true;
                        }
                    }
                    
                    const radioOrCheckbox = el.querySelector('input[type="radio"], input[type="checkbox"]');
                    if (radioOrCheckbox) {
                        return true;
                    }
                }
                
                const parent = el.parentElement;
                if (parent) {
                    const hasRadioSibling = parent.querySelector('input[type="radio"], input[type="checkbox"]');
                    if (hasRadioSibling) {
                        if (!parent.className.includes('question-container') && 
                            !parent.className.includes('form-group')) {
                            return true;
                        }
                    }
                }
                
                const role = el.getAttribute('role');
                if (role === 'radio' || role === 'checkbox' || role === 'option') {
                    return true;
                }
                
                return false;
            };
            
            const getElementPriority = (el) => {
                const tag = el.tagName.toLowerCase();
                const className = (el.className || '').toString().toLowerCase();
                
                if (className.includes('question') || className.includes('prompt')) {
                    return 1;
                }
                
                if (['h1', 'h2', 'h3'].includes(tag)) {
                    return 2;
                }
                
                if (['h4', 'h5', 'h6', 'p'].includes(tag)) {
                    return 3;
                }
                
                return 4;
            };
            
            const xpathQueries = [
                '//h1 | //h2 | //h3 | //h4 | //h5 | //h6',
                '//p',
                '//span[not(@data-test-id="user-balance")]',
                '//div[contains(@class, "question")]',
                '//label',
                '//li',
                '//td | //th',
                '//*[@role="heading"]',
                '//*[contains(@class, "question-title")]'
            ];
            
            xpathQueries.forEach(xpath => {
                try {
                    const result = document.evaluate(
                        xpath,
                        document,
                        null,
                        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                        null
                    );
                    
                    for (let i = 0; i < result.snapshotLength; i++) {
                        const el = result.snapshotItem(i);
                        
                        if (isHoneypot(el)) {
                            continue;
                        }
                        
                        if (isRadioOrCheckboxText(el)) {
                            continue;
                        }
                        
                        
                        if (!isElementReallyVisible(el)) {
                            continue;
                        }
                        
                        let text = (el.innerText || el.textContent || '').trim();
                        
                        if (!text && el.getAttribute) {
                            text = el.getAttribute('aria-label') || '';
                        }
                        
                        text = text.trim();
                        
                        if (text.match(/\$\s*[\d,]+\.\d{2}\s*(USD|EUR|GBP)/)) {
                            continue;
                        }
                        
                        if (text.length < 2) {
                            continue;
                        }
                        
                        if (text && !processedTexts.has(text)) {
                            processedTexts.add(text);
                            texts.push(text);
                            
                            const rect = el.getBoundingClientRect();
                            const tag = el.tagName.toLowerCase();
                            const className = (el.className || '').toString();
                            const style = getComputedStyle(el);
                            const sh = el.scrollHeight || 0;
                            const ch = el.clientHeight || 0;
                            const sw = el.scrollWidth || 0;
                            const cw = el.clientWidth || 0;
                            const overflowY = (style.overflowY || '').toLowerCase();
                            const overflowX = (style.overflowX || '').toLowerCase();
                            const scrollableY = (sh - ch) > 8 && (overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay' || overflowY === '');
                            const scrollableX = (sw - cw) > 8 && (overflowX === 'auto' || overflowX === 'scroll' || overflowX === 'overlay');
                            
                            elements.push({
                                text: text.substring(0, 500),
                                tag: tag,
                                className: className,
                                id: el.id || null,
                                priority: getElementPriority(el),
                                bbox: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    width: Math.round(rect.width),
                                    height: Math.round(rect.height),
                                    center_x: Math.round(rect.left + rect.width / 2),
                                    center_y: Math.round(rect.top + rect.height / 2),
                                    right: Math.round(rect.right),
                                    bottom: Math.round(rect.bottom)
                                },
                                isQuestion: className.includes('question') || className.includes('prompt'),
                                scrollable: scrollableY || scrollableX,
                                scrollableY: scrollableY,
                                scrollableX: scrollableX
                            });
                        }
                    }
                } catch (e) {
                    // Ignoruj bĹ‚Ä™dy XPath
                }
            });
            
            const modal = document.querySelector('[role="dialog"]:not([aria-hidden="true"])');
            if (modal) {
                const modalXpath = './/p | .//h1 | .//h2 | .//h3 | .//span | .//div[contains(@class, "question")]';
                const modalResult = document.evaluate(
                    modalXpath,
                    modal,
                    null,
                    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                    null
                );
                
                for (let i = 0; i < modalResult.snapshotLength; i++) {
                    const el = modalResult.snapshotItem(i);
                    
                    if (isHoneypot(el) || isRadioOrCheckboxText(el) || isClickable(el)) continue;
                    
                    const text = (el.innerText || el.textContent || '').trim();
                    
                    if (text && text.length > 1 && !processedTexts.has(text)) {
                        processedTexts.add(text);
                        if (!texts.includes(text)) {
                            texts.push(text);
                            
                            const rect = el.getBoundingClientRect();
                            const tag = el.tagName.toLowerCase();
                            const className = (el.className || '').toString();
                            const style = getComputedStyle(el);
                            const sh = el.scrollHeight || 0;
                            const ch = el.clientHeight || 0;
                            const sw = el.scrollWidth || 0;
                            const cw = el.clientWidth || 0;
                            const overflowY = (style.overflowY || '').toLowerCase();
                            const overflowX = (style.overflowX || '').toLowerCase();
                            const scrollableY = (sh - ch) > 8 && (overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay' || overflowY === '');
                            const scrollableX = (sw - cw) > 8 && (overflowX === 'auto' || overflowX === 'scroll' || overflowX === 'overlay');
                            
                            elements.push({
                                text: text.substring(0, 500),
                                tag: tag,
                                className: className,
                                id: el.id || null,
                                priority: getElementPriority(el),
                                bbox: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    width: Math.round(rect.width),
                                    height: Math.round(rect.height),
                                    center_x: Math.round(rect.left + rect.width / 2),
                                    center_y: Math.round(rect.top + rect.height / 2),
                                    right: Math.round(rect.right),
                                    bottom: Math.round(rect.bottom)
                                },
                                isQuestion: className.includes('question') || className.includes('prompt'),
                                scrollable: scrollableY || scrollableX,
                                scrollableY: scrollableY,
                                scrollableX: scrollableX,
                                inModal: true
                            });
                        }
                    }
                }
            }
            
            elements.sort((a, b) => {
                if (a.priority !== b.priority) {
                    return a.priority - b.priority;
                }
                return a.bbox.y - b.bbox.y;
            });
            
            return {
                text: texts.join('\n'),
                elements: elements
            };
        }
        """
        
        try:
            result = await self.safe_eval(js, default={"text": "", "elements": []}, page=page)
            text = result.get("text", "")
            elements = result.get("elements", [])
            if not text:
                js_fb = r"""
                () => {
                  const out = { text: '', elements: [] };
                  const pushEl = (el) => {
                    try {
                      const r = el.getBoundingClientRect();
                      if (!r || r.width <= 0 || r.height <= 0) return;
                      const t = (el.innerText || el.textContent || '').trim();
                      if (!t) return;
                      out.elements.push({
                        text: t,
                        tag: (el.tagName || '').toLowerCase(),
                        bbox: { x: Math.round(r.left), y: Math.round(r.top), width: Math.round(r.width), height: Math.round(r.height) },
                        priority: 3
                      });
                    } catch(e){}
                  };
                  const collect = (root) => {
                    try {
                      const doc = root && (root.nodeType === 9 ? root : root.ownerDocument || document);
                      if (!doc) return;
                      const body = (root.body) ? root.body : (doc.body || null);
                      if (body) {
                        const bt = (body.innerText || body.textContent || '').trim();
                        if (bt) out.text += bt + '\n';
                      }
                      const picks = (root.querySelectorAll ? root.querySelectorAll('h1,h2,h3,[role="heading"],header,[data-title]') : []);
                      if (picks && picks.length){ picks.forEach(pushEl); }
                      if (root.querySelectorAll) {
                        root.querySelectorAll('*').forEach(el => { try { if (el.shadowRoot) collect(el.shadowRoot); } catch(e){} });
                      }
                    } catch(e) {}
                  };
                  try { collect(document); } catch(e) {}
                  try {
                    document.querySelectorAll('iframe').forEach(ifr => {
                      try { if (ifr && ifr.contentDocument) collect(ifr.contentDocument); } catch(e){}
                    });
                  } catch(e) {}
                  out.text = out.text.trim();
                  return out;
                }
                """
                fb = await self.safe_eval(js_fb, default={"text": "", "elements": []}, page=page)
                if fb and isinstance(fb, dict):
                    text = fb.get("text", "")
                    elements = fb.get("elements", [])
            if text:
                log(f"đź“ť XPath extraction: {len(text)} chars, {len(elements)} elements", "INFO")
            else:
                log("âš ď¸Ź No visible text found!", "WARNING")
                
            return text, elements
        except Exception as e:
            log(f"âťŚ XPath extraction error: {e}", "ERROR")
            return "", []

    def _get_ocr_text(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Zwraca tekst z OCR wraz z koordynatami."""
        if not self.ocr_lines_filtered:
            return "", []
        
        ocr_texts = []
        ocr_elements = []
        
        for line in self.ocr_lines_filtered:
            if line.text and len(line.text.strip()) > 1:
                ocr_texts.append(line.text.strip())
                ocr_elements.append({
                    "text": line.text.strip(),
                    "source": "OCR",
                    "confidence": line.conf,
                    "bbox": line.bbox,
                    "priority": 5
                })
        
        button_texts = set()
        for c in self.clickables:
            if c.attributes.get("is_clickable", False) and c.text:
                button_texts.add(c.text.lower().strip())
        
        filtered_ocr = []
        filtered_elements = []
        
        for i, text in enumerate(ocr_texts):
            text_lower = text.lower().strip()
            if text_lower not in button_texts:
                filtered_ocr.append(text)
                filtered_elements.append(ocr_elements[i])
        
        return '\n'.join(filtered_ocr), filtered_elements

    async def extract_question_text(self, page=None) -> Tuple[str, List[Dict[str, Any]]]:
        """POPRAWIONA - inteligentniejsza ekstrakcja pytaĹ„"""
        p = page or self.page
        if not p or p.is_closed():
            return "", []

        try:
            _, dom_elements = await self._extract_dom_text(page)
            
            # Filtruj clickables
            clickable_texts = {norm_text(c.text) for c in self.clickables if c.text}
            
            # NOWE: Priorytetyzuj elementy ktĂłre wyglÄ…dajÄ… jak pytania
            question_indicators = [
                '?',  # Znak zapytania
                'proszÄ™', 'please',
                'wybierz', 'select', 'choose',
                'podaj', 'enter', 'provide',
                'jaki', 'ktĂłry', 'ile', 'czy',
                'what', 'which', 'how', 'when', 'where', 'why'
            ]
            
            question_elements = []
            for elem in dom_elements:
                text = norm_text(elem.get("text", ""))
                if not text or text in clickable_texts:
                    continue
                
                # SprawdĹş czy to pytanie
                is_question = False
                text_lower = text.lower()
                
                # Ma znak zapytania lub keyword
                for indicator in question_indicators:
                    if indicator in text_lower:
                        is_question = True
                        elem["priority"] = 1  # Wysoki priorytet
                        break
                
                # MoĹĽe byÄ‡ pytaniem jeĹ›li jest w okreĹ›lonych tagach
                tag = elem.get("tag", "").lower()
                if tag in ['h1', 'h2', 'h3', 'legend', 'label']:
                    if not is_question:
                        elem["priority"] = 2
                    is_question = True
                
                # Pomijaj elementy ktĂłre na pewno NIE sÄ… pytaniami
                if any(skip in text_lower for skip in ['cookie', 'privacy', 'terms', 'Â©', 'copyright']):
                    continue
                
                if is_question or elem.get("priority", 99) <= 3:
                    question_elements.append(elem)
            
            # Sortuj inteligentnie
            question_elements.sort(key=lambda x: (
                x.get("priority", 99),
                x.get("bbox", {}).get("y", 0),  # Od gĂłry
                -len(x.get("text", ""))  # DĹ‚uĹĽsze pierwsze
            ))
            
            # Ogranicz do sensownej liczby (nie caĹ‚Ä… stronÄ™!)
            question_elements = question_elements[:10]
            
            final_text = "\n".join([elem.get("text", "") for elem in question_elements])
            
            log(f"đź“ť Question: {len(final_text)} chars from {len(question_elements)} elements", "INFO")
            
            if not self.page.is_closed():
                await self._save_question_json(final_text, ["DOM_FILTERED_SMART"], question_elements)
            
            return final_text, question_elements
            
        except Exception as e:
            log(f"extract_question_text error: {e}", "ERROR")
            return "", []

    async def _save_question_json(self, text: str, sources: List[str], elements: List[Dict[str, Any]]):
        """Zapisuje tekst i koordynaty do current_question.json."""
        question_file = self.output_dir / "current_question.json"
        
        page_url = ""
        page_title = ""
        active_area_type = "unknown"
        viewport = {"width": 0, "height": 0}
        
        if self.page and not self.page.is_closed():
            with contextlib.suppress(Exception):
                page_url = self.page.url
                page_title = await self.page.title()
                viewport = await self.safe_eval("""() => ({ width: window.innerWidth, height: window.innerHeight })""", default={"width": 0, "height": 0})
                has_modal = await self.safe_eval("""
                    () => {
                        const modals = document.querySelectorAll('[role="dialog"]:not([aria-hidden="true"]), .modal.show, [aria-modal="true"]');
                        for (const m of modals) {
                            const style = window.getComputedStyle(m);
                            if (style.display !== 'none' && style.visibility !== 'hidden') return true;
                        }
                        return false;
                    }
                """, default=False)
                active_area_type = "modal" if has_modal else "viewport"
        
        main_question = None
        question_elements = []
        for elem in elements:
            if elem.get("isQuestion", False) or "question" in elem.get("className", "").lower():
                question_elements.append(elem)
                if not main_question or elem.get("priority", 99) < main_question.get("priority", 99):
                    main_question = elem
        if not main_question and elements:
            for elem in elements:
                if elem.get("priority", 99) <= 2:
                    main_question = elem
                    break
        
        question_data = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "url": page_url,
            "title": page_title,
            "full_text": text if text else "",
            "text_length": len(text) if text else 0,
            "sources": sources if sources else [],
            "area_type": active_area_type,
            "viewport": viewport,
            "clickables_count": len(self.clickables),
            "lines_count": len(text.split('\n')) if text else 0,
            "words_count": len(text.split()) if text else 0,
            "ocr_available": len(self.ocr_lines_filtered) > 0,
            "has_content": len(text) > 0,
            "main_question": {
                "text": main_question.get("text", "") if main_question else "",
                "bbox": main_question.get("bbox", {}) if main_question else {},
                "tag": main_question.get("tag", "") if main_question else "",
                "className": main_question.get("className", "") if main_question else ""
            } if main_question else None,
            "question_elements": question_elements,
            "text_elements": elements[:50]
        }
        
        with contextlib.suppress(Exception):
            with open(question_file, "w", encoding="utf-8") as f:
                json.dump(question_data, f, ensure_ascii=False, indent=2)
            if text and len(text) > 0:
                if main_question and main_question.get('bbox'):
                    bq = main_question['bbox']
                    log(f"âś… Question saved ({len(text)} chars) | Main Q at ({bq.get('x')},{bq.get('y')})", "SUCCESS")
                else:
                    log(f"âś… Question saved ({len(text)} chars) | No main question detected", "SUCCESS")

    async def get_active_area(self) -> Dict[str, Any]:
        """Zwraca wspĂłĹ‚rzÄ™dne aktywnego obszaru + rozmiar viewportu i dpr (do OCR)."""
        js = r"""
        () => {
            const modalSelectors = [
                '[role="dialog"]:not([aria-hidden="true"])',
                '[role="alertdialog"]',
                '[aria-modal="true"]',
                '.modal.show',
                '.modal.is-active',
                '.modal.open',
                '.popup:not(.hidden)',
                'div[class*="modal"]:not(.hidden)',
                'div[class*="popup"]:not(.hidden)',
                'div[class*="dialog"]:not(.hidden)',
                '.ReactModal__Content',
                '.MuiDialog-root',
                '.ant-modal-content'
            ];
            let activeModal = null;
            let highestZ = -1;
            document.querySelectorAll(modalSelectors.join(',')).forEach(modal => {
                const style = window.getComputedStyle(modal);
                if (style.display === 'none' || style.visibility === 'hidden') return;
                const rect = modal.getBoundingClientRect();
                if (rect.width < 50 || rect.height < 50) return;
                const z = parseInt(style.zIndex) || 0;
                if (z > highestZ || !activeModal) {
                    highestZ = z;
                    activeModal = modal;
                }
            });
            const common = { dpr: window.devicePixelRatio || 1, vw: window.innerWidth, vh: window.innerHeight };
            if (activeModal) {
                const content = activeModal.querySelector('.modal-content, .modal-body, [class*="content"], [class*="body"]') || activeModal;
                const r = content.getBoundingClientRect();
                return Object.assign({
                    type: 'modal',
                    x: Math.max(0, Math.round(r.left)),
                    y: Math.max(0, Math.round(r.top)),
                    width: Math.round(r.width),
                    height: Math.round(r.height),
                    right: Math.round(r.right),
                    bottom: Math.round(r.bottom)
                }, common);
            }
            return Object.assign({
                type: 'viewport',
                x: 0, y: 0,
                width: window.innerWidth,
                height: window.innerHeight,
                right: window.innerWidth,
                bottom: window.innerHeight
            }, common);
        }
        """
        try:
            area = await self.safe_eval(js, default=None)
            if area:
                log(f"đź“ Active area: {area['type']} {area['width']}x{area['height']} (vw={area['vw']} vh={area['vh']} dpr={area['dpr']})", "DEBUG")
            return area or {"type": "viewport", "x": 0, "y": 0, "width": 1920, "height": 1080, "vw": 1920, "vh": 1080, "dpr": 1}
        except Exception:
            return {"type": "viewport", "x": 0, "y": 0, "width": 1920, "height": 1080, "vw": 1920, "vh": 1080, "dpr": 1}

    def _ocr_sync(self, img_bytes: bytes, active_area: Dict[str, Any] = None) -> Tuple[List[OcrLine], List[OcrLine]]:
        return [], []


    async def _do_ocr_async(self, shot_bytes: bytes):
        return


    def _match_tab_by_ocr(self, ocr_text: str, win_info: Optional[WindowInfo]) -> Dict[str, Any]:
        """POPRAWIONA - szybsze matching"""
        if not win_info or not sys.platform.startswith("win"):
            return {}
        
        # Ogranicz dĹ‚ugoĹ›Ä‡ do 200 znakĂłw
        ocr_text_short = ocr_text[:200]
        title_short = win_info.title[:200]
        
        # Szybkie sprawdzenie wspĂłlnych sĹ‚Ăłw (zamiast peĹ‚nego fuzzy)
        ocr_words = set(ocr_text_short.lower().split())
        title_words = set(title_short.lower().split())
        
        if ocr_words and title_words:
            common = len(ocr_words & title_words)
            total = len(ocr_words | title_words)
            score = common / total if total else 0
        else:
            score = 0
        
        urls = extract_url_candidates(ocr_text)
        
        return {
            "window_title": win_info.title,
            "match_score": round(score, 3),
            "ocr_urls": urls[:3],  # Ogranicz
            "process": win_info.process_name,
        }

    async def _do_snapshot(self):
        """POPRAWIONA - modularna z losowoĹ›ciÄ…"""
        start_time = self.perf.start("snapshot")
        
        # Trzymaj zgodność ze "star" (active_page_id):
        # najpierw reselect, potem weź aktywną gwiazdkę; dopiero na końcu fallback na most-visible
        with contextlib.suppress(Exception):
            await self._reselect_active_page()
        active_page = await self._get_active_page()
        if not active_page:
            active_page = await self._get_most_visible_page()
        if not active_page or active_page.is_closed():
            self.perf.end("snapshot", start_time)
            return
        # Ensure all downstream helpers use the right page object
        if getattr(self, 'page', None) is None or self.page is not active_page:
            self.page = active_page
            with contextlib.suppress(Exception):
                self.active_page_id = str(id(active_page))
        # Informacyjny log, którą stronę skanujemy (do pliku i konsoli)
        try:
            url_now = getattr(active_page, 'url', '')
            log(f"Snapshot page -> {url_now[:160]}", "INFO")
        except Exception:
            pass
        # Refresh quick state and always get minimal page_data first
        page_data = None
        try:
            pid = str(id(active_page))
            tr = getattr(self, 'tracked_pages', {}).get(pid)
            if tr:
                with contextlib.suppress(Exception):
                    await self._page_state_quick(tr)
            page_data = await self._collect_page_data(active_page)
            if tr and isinstance(page_data, dict):
                url_pd = page_data.get("url")
                title_pd = page_data.get("title")
                if url_pd:
                    tr.url = url_pd
                if title_pd:
                    tr.title = title_pd
            # Scan strictly the currently visible tab (active in the window)
            if tr and (not getattr(tr, 'is_visible', False)):
                # Update page info then skip heavy ops
                with contextlib.suppress(Exception):
                    await self._save_page_info(page_data or {})
                self.perf.end("snapshot", start_time)
                return
        except Exception:
            pass
        
        # === LOSOWA KOLEJNOĹšÄ† OPERACJI (bardziej ludzkie) ===
        operations = [
            ("page_data", self._collect_page_data),
            ("html", self._collect_html),
            ("clickables", self._collect_clickables),
            ("question", self._collect_question),
        ]
        
        # Czasem (30%) zmieĹ„ kolejnoĹ›Ä‡
        if random.random() < 0.3:
            random.shuffle(operations)
        
        results = {}
        for name, func in operations:
            if active_page.is_closed():
                break
            
            # MaĹ‚e opĂłĹşnienie miÄ™dzy operacjami
            if results:  # Nie przed pierwszÄ…
                await asyncio.sleep(random.uniform(0.01, 0.05))
            
            try:
                results[name] = await func(active_page)
            except Exception as e:
                log(f"Snapshot {name} error: {e}", "DEBUG")
                results[name] = None
        
        # === WARUNKOWY ZAPIS (nie zawsze wszystko!) ===
        
        # Zawsze zapisz snapshot (deterministyczny output)
        await self._save_snapshot(results)
        
        # 30% zrób screenshot (wyłączone w trybie DOM-only)
                
        # Zawsze zapisz page info (prefer collected page_data)
        await self._save_page_info((results.get("page_data") or page_data or {}) if isinstance((results.get("page_data") or page_data), dict) else {})
        
        # Zawsze stats
        self.stats["total_snapshots"] += 1
        self.perf.end("snapshot", start_time)
        
        # GC z losowoĹ›ciÄ…
        if self.stats["total_snapshots"] % random.randint(40, 60) == 0:
            gc.collect()

    async def _collect_page_data(self, page):
        """Osobna funkcja do page data"""
        script = """() => ({
            url: window.location.href,
            title: document.title,
            viewport: {width: window.innerWidth||0, height: window.innerHeight||0},
            hasFocus: document.hasFocus(),
            visibility: document.visibilityState
        })"""
        return await self.safe_eval(script, default={}, page=page)

    async def _collect_html(self, page):
        """Osobna funkcja do HTML"""
        return await self.safe_content(default="", page=page)

    async def _collect_clickables(self, page):
        """Osobna funkcja do clickables"""
        self.clickables = await self.extract_clickables(page)
        return self.clickables

    async def _collect_question(self, page):
        """Osobna funkcja do question"""
        text, elements = await self.extract_question_text(page)
        return {"text": text, "elements": elements}

    async def _save_files_async(self, snapshot: Dict):
        loop = asyncio.get_running_loop()

        def write_files():
            try:
                snapshot_out = dict(snapshot or {})
                clickables_for_file = []
                try:
                    raw_clicks = snapshot_out.get("clickables")
                    if raw_clicks is not None:
                        clickables_serial = []
                        for c in (raw_clicks or []):
                            try:
                                d = asdict(c)
                            except Exception:
                                d = c if isinstance(c, dict) else None
                            if d is None:
                                continue
                            if "label_text" not in d:
                                d["label_text"] = None
                            clickables_serial.append(d)
                        snapshot_out["clickables"] = clickables_serial
                        clickables_for_file = clickables_serial[:200]
                    else:
                        snapshot_out["clickables"] = []
                except Exception:
                    snapshot_out["clickables"] = []
                    clickables_for_file = []

                snapshot_json = json.dumps(snapshot_out, separators=(",", ":"))
                snap_hash = md5(snapshot_json)
                now = time.time()

                skip_snapshot = (
                    snap_hash == getattr(self, "last_snapshot_hash", "")
                    and (now - float(getattr(self, "last_write_time", 0.0))) < self.save_min_interval
                )

                if not skip_snapshot:
                    with open(self.file_snapshot, "w", encoding="utf-8") as f:
                        f.write(snapshot_json)
                        json.dump(snapshot_out, f, separators=(",", ":"))
                    self.last_snapshot_hash = snap_hash
                    self.last_write_time = now

                with open(self.file_clickables, "w", encoding="utf-8") as f:
                    json.dump(clickables_for_file, f, separators=(",", ":"))

                with open(self.file_stats, "w", encoding="utf-8") as f:
                    json.dump(self.stats, f, separators=(",", ":"))

                if not skip_snapshot:
                    self.stats["files_written"] += 1
                return True
            except Exception as e:
                log(f"��� Write error: {e}", "ERROR")
                return False

        await loop.run_in_executor(None, write_files)

    async def _save_page_info(self, page_data: Dict):
        """Zapisz lekkie info o stronie na podstawie page_data."""
        ph: Optional[str] = None
        try:
            url = page_data.get("url", "") if isinstance(page_data, dict) else ""
            title = page_data.get("title", "") if isinstance(page_data, dict) else ""
            info = {
                "timestamp": time.time(),
                "url": url,
                "title": title,
                "interactive_elements_count": len(self.clickables),
                "page_state": getattr(self, "page_state", "idle"),
                "tracking": {
                    "active_page_id": getattr(self, "active_page_id", None),
                    "has_focus": page_data.get("hasFocus", False) if isinstance(page_data, dict) else False,
                    "visibility": page_data.get("visibility", "unknown") if isinstance(page_data, dict) else "unknown",
                },
            }
            payload = json.dumps(info, separators=(",", ":"))
            loop = asyncio.get_running_loop()
            ph = md5(payload)
            now = time.time()
            if ph == getattr(self, '_last_page_md5', '') and (now - float(getattr(self, 'last_write_time', 0.0)) < self.save_min_interval):
                with contextlib.suppress(Exception):
                    self._write_tabs_file()
                return
            def _write_page_files() -> None:
                try:
                    self.file_page.write_text(payload, encoding="utf-8")
                except Exception:
                    pass
                try:
                    self.file_page_static.write_text(payload, encoding="utf-8")
                except Exception:
                    pass

            await loop.run_in_executor(None, _write_page_files)
            self._last_page_md5 = ph
            with contextlib.suppress(Exception):
                self._write_tabs_file()
        except Exception as e:
            if ph is not None:
                self._last_page_md5 = ph
            log(f"save_page_info error: {e}", "DEBUG")

    async def _save_snapshot(self, snapshot: Dict):
        """Zapisz główny snapshot oraz pliki towarzyszące."""
        try:
            await self._save_files_async(snapshot)
            self.last_successful_operation = time.time()
        except Exception as e:
            log(f"snapshot save error: {e}", "DEBUG")

    async def _watchdog_loop(self):
        while self.recording:
            try:
                if time.time() - self.last_successful_operation > WATCHDOG_TIMEOUT:
                    log(f"đź¶ Watchdog: no successful op for {WATCHDOG_TIMEOUT}s", "WARNING")
                    self.stats["watchdog_resets"] += 1
                    self.force_next_snapshot = True
                    if self.page and not self.page.is_closed():
                        with contextlib.suppress(Exception):
                            await self.safe_eval("() => document.visibilityState", default=None)

                # OdĹ›wieĹĽ stany tracked pages (nieinwazyjnie)
                for pid, tracker in list(self.tracked_pages.items()):
                    if tracker.page.is_closed():
                        self.tracked_pages.pop(pid, None)
                        continue
                    with contextlib.suppress(Exception):
                        await self._page_state_quick(tracker)

                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.verbose:
                    log(f"Watchdog error: {e}", "DEBUG")
                await asyncio.sleep(3.0)














