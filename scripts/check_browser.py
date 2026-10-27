"""Start a temporary local API, run Playwright checks, then stop the API."""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    output = Path('.ui-qa')
    output.mkdir(exist_ok=True)
    with (output / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                                   '--host', '127.0.0.1', '--port', '8011'],
                                  stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urlopen('http://127.0.0.1:8011/health', timeout=1):
                        break
                except OSError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError((output / 'server.log').read_text())
                    time.sleep(.2)
            env = dict(os.environ, E2E_BASE_URL='http://127.0.0.1:8011')
            return subprocess.run(['node', 'tests/browser.cjs'], env=env, timeout=120).returncode
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    sys.exit(main())
