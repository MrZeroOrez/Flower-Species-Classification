// Start the API with real weights first; run npm run test:e2e.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const base = process.env.E2E_BASE_URL || 'http://127.0.0.1:8000';
  const manifest = JSON.parse(fs.readFileSync('data/manifest.json', 'utf8'));
  const sample = manifest.records.find(row => row.split === 'test');
  const image = process.env.E2E_IMAGE || path.join('data', sample.path);
  const importedRuntime = process.env.E2E_BROWSER_MODULE ? require(process.env.E2E_BROWSER_MODULE) : null;
  const runtime = importedRuntime?.default || importedRuntime;
  const browser = await chromium.launch(runtime
    ? { headless: true, executablePath: process.env.E2E_BROWSER_EXECUTABLE || await runtime.executablePath(), args: runtime.args }
    : { headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', failure => errors.push(failure.message));
  fs.mkdirSync('.ui-qa', { recursive: true });
  try {
    await page.goto(base);
    await page.getByText('Model ready', { exact: true }).waitFor();
    assert.equal(await page.locator('#class-list span').count(), manifest.classes.length);
    assert.equal(await page.locator('#identify-button').isDisabled(), true);
    await page.screenshot({ path: '.ui-qa/desktop-empty.png', fullPage: true });

    await page.locator('#image-file').setInputFiles(image);
    await page.waitForFunction(() => !document.getElementById('identify-button').disabled);
    assert.equal(await page.locator('#preview').isVisible(), true);
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/predict') && r.request().method() === 'POST');
    await page.locator('#identify-button').click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    const result = await response.json();
    await page.locator('#result-content').waitFor({ state: 'visible' });
    assert.equal((await page.locator('#predicted-class').textContent()).toLowerCase(), result.predicted_class);
    assert.equal(await page.locator('#primary-score').textContent(), `${(result.score * 100).toFixed(1)}%`);
    assert.equal(await page.locator('#prediction-list li').count(), 3);
    await page.screenshot({ path: '.ui-qa/desktop-result.png', fullPage: true });

    await page.getByRole('button', { name: 'Clear image' }).click();
    assert.equal(await page.locator('#preview').isVisible(), false);
    assert.equal(await page.locator('#result-content').isVisible(), false);
    assert.equal(await page.locator('#identify-button').isDisabled(), true);
    await page.locator('#image-file').setInputFiles({ name: 'bad.png', mimeType: 'image/png', buffer: Buffer.from('broken') });
    await page.getByRole('alert').filter({ hasText: 'could not be opened' }).waitFor();
    assert.equal(await page.locator('#identify-button').isDisabled(), true);
    await page.locator('#image-file').setInputFiles({ name: 'large.png', mimeType: 'image/png', buffer: Buffer.alloc(5 * 1024 * 1024 + 1) });
    await page.getByRole('alert').filter({ hasText: '5 MiB or smaller' }).waitFor();

    await page.locator('#image-file').setInputFiles(image);
    await page.waitForFunction(() => !document.getElementById('identify-button').disabled);
    await page.route('**/predict', route => route.fulfill({ status: 500, json: { detail: 'Prediction failed. Please try again.' } }));
    await page.locator('#identify-button').click();
    await page.getByRole('alert').filter({ hasText: 'Prediction failed' }).waitFor();
    assert.equal(await page.locator('#identify-button').isEnabled(), true);
    await page.unroute('**/predict');

    // Reset while a request is in flight must not display the stale response.
    let release;
    const held = new Promise(resolve => { release = resolve; });
    await page.route('**/predict', async route => { await held; await route.fulfill({ json: result }).catch(() => {}); });
    const requestStarted = page.waitForRequest('**/predict');
    await page.locator('#identify-button').click();
    await requestStarted;
    await page.getByRole('button', { name: 'Clear image' }).click();
    release();
    await page.unroute('**/predict', undefined, { behavior: 'wait' });
    assert.equal(await page.locator('#result-content').isVisible(), false);
    assert.equal(await page.locator('#result-panel').getAttribute('aria-busy'), 'false');

    await page.route('**/health', route => route.fulfill({ json: { status: 'ok', model_ready: false } }));
    await page.reload();
    await page.getByText('Model unavailable', { exact: true }).waitFor();
    assert.equal(await page.locator('#service-message').isVisible(), true);
    assert.equal(await page.locator('#identify-button').isDisabled(), true);
    await page.unroute('**/health');
    await page.getByRole('button', { name: 'Check again' }).click();
    await page.getByText('Model ready', { exact: true }).waitFor();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#image-file').setInputFiles(image);
    await page.waitForFunction(() => !document.getElementById('identify-button').disabled);
    await page.locator('#identify-button').click();
    await page.locator('#result-content').waitFor({ state: 'visible' });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: '.ui-qa/mobile-result.png', fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS: real image → API → model → rendered scores; reset; invalid/oversize upload; server failure; stale-response cancellation; missing model/retry; mobile layout; no browser errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
