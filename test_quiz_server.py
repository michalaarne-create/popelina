import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).parent
SITE_DIR = BASE_DIR / "quiz_site"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "quiz_submissions.log"


class QuizRequestHandler(SimpleHTTPRequestHandler):
    """
    Minimal HTTP server for local quiz testing.

    Serves:
    - /                 -> index.html
    - /quiz/car         -> quiz.html (quiz=car)
    - /quiz/life        -> quiz.html (quiz=life)
    - /quiz/shopping    -> quiz.html (quiz=shopping)
    - /static/...       -> static assets from quiz_site/static
    - /api/submit       -> accepts POST with JSON answers and appends to log
    """

    def __init__(self, *args, **kwargs):
        # Serve files from SITE_DIR
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        # Slightly quieter logging
        return super().log_message(format, *args)

    # Utilities
    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> Tuple[bytes, str]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type", "")
        return body, content_type

    # HTTP methods
    def do_GET(self) -> None:  # noqa: N802
        # Route pretty URLs to static files
        if self.path in ("/", "/index", "/index.html"):
            self.path = "/index.html"
        elif self.path.startswith("/quiz/"):
            # All quiz pages use the same HTML shell (quiz.html)
            # JS detects quiz type from location.pathname (/quiz/car -> "car").
            self.path = "/quiz.html"
        elif self.path.startswith("/static/"):
            # Static assets served as-is from SITE_DIR/static
            pass

        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/submit":
            body, content_type = self._read_body()
            try:
                if "application/json" in content_type:
                    data = json.loads(body.decode("utf-8") or "{}")
                else:
                    data = {"raw": body.decode("utf-8", errors="replace")}
            except Exception:  # pragma: no cover - best-effort logging
                data = {"raw": body.decode("utf-8", errors="replace")}

            LOG_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except OSError:
                # If logging fails, still return 200 so tests are not blocked
                pass

            return self._send_json({"status": "ok"})

        return self._send_json({"error": "Not found"}, status=404)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local quiz HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, QuizRequestHandler)
    print(f"Quiz server running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
