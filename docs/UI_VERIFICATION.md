# Upload and prediction UI verification

Story: choose a flower photo on the home page -> submit multipart file to
POST /predict -> run the exported CPU ResNet9 -> render the class and top-three
model scores in the browser.

## Verified boundaries

| Boundary | Evidence |
|---|---|
| UI and static assets | Home page, app.js and styles.css returned HTTP 200 |
| Model readiness | GET /health and /classes returned 200; UI showed Model ready and five categories |
| Browser to API | Real held-out JPEG submitted to POST /predict; HTTP 200 |
| Model to UI | Daisy appeared with 96.5% uncalibrated model score; rendered score matched response |
| Invalid inputs | Corrupt and >5 MiB uploads showed errors and disabled submission |
| API failure | Simulated 500 displayed error and allowed retry |
| Selection change | Clearing during a pending request prevented stale results |
| Missing artifacts | Simulated unavailable model disabled prediction, displayed recovery message; retry restored readiness |
| Mobile | 390px viewport had no horizontal overflow and completed a real prediction |
| Browser errors | No uncaught page errors in the successful suite |

Desktop and mobile screenshots were inspected locally. They are not application
assets or distributed artifacts. The browser suite uses actual trained weights
for successful inference; failure states use explicitly intercepted responses.

## Checks

- 13 Python API/data tests.
- JavaScript syntax validation.
- Playwright/Chromium suite: real upload, scores, reset, invalid/oversized images,
  server failure, stale-response cancellation, missing-model retry and mobile.

The optional browser checks require Node/Playwright plus dataset and model
artifacts. Normal application startup only requires the Python dependencies.
The verification environment used a packaged Chromium binary because the normal
browser download endpoint did not provide a usable archive. This workaround is
not an application dependency.

## Scope and limitations

The UI does not add care advice, agent workflows, user accounts or memory.
Evaluation remains the existing three-epoch baseline: 67.70% held-out accuracy,
28.21% rose recall, uncalibrated scores and no unsupported-image rejection.
This verifies local behavior, not an externally deployed service or free-host
resource limits.
