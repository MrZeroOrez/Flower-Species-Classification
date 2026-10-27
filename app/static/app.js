'use strict';
const $ = id => document.getElementById(id);
const state = { file: null, previewUrl: null, modelReady: false, busy: false, version: 0, controller: null };
const titleCase = value => value.charAt(0).toUpperCase() + value.slice(1);
function updateButton() { $('identify-button').disabled = !state.file || !state.modelReady || state.busy; }
function error(message = '') { $('upload-error').textContent = message; $('upload-error').hidden = !message; }
function resetResult() { $('result-content').hidden = true; $('result-empty').hidden = false; $('result-announcement').textContent = ''; resetCare(); }
function cancelRequest() {
  state.version++;
  state.controller?.abort(); state.controller = null; state.busy = false;
  $('button-label').textContent = 'Identify flower'; $('result-panel').setAttribute('aria-busy', 'false');
}
function clearSelection() {
  cancelRequest();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null; state.file = null;
  $('image-file').value = ''; $('preview').removeAttribute('src'); $('preview').hidden = true;
  $('upload-empty').hidden = false; $('clear-button').hidden = true;
  $('file-name').textContent = 'No image selected'; error(); resetResult(); updateButton();
}
async function selectFile(file) {
  clearSelection();
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) return error('Choose an image that is 5 MiB or smaller.');
  if (!file.size) return error('This file is empty. Choose a JPEG or PNG image.');
  if (!['image/jpeg', 'image/png'].includes(file.type) && !(file.type === '' && /\.(jpe?g|png)$/i.test(file.name))) return error('Choose a JPEG or PNG image.');
  const version = state.version;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (version !== state.version) { URL.revokeObjectURL(url); return; }
    if (image.naturalWidth * image.naturalHeight > 12000000) { URL.revokeObjectURL(url); return error('Choose an image with 12 megapixels or fewer.'); }
    state.previewUrl = url; state.file = file;
    $('preview').src = url; $('preview').hidden = false; $('upload-empty').hidden = true;
    $('clear-button').hidden = false; $('file-name').textContent = file.name; updateButton();
  } catch {
    URL.revokeObjectURL(url);
    if (version === state.version) error('This image could not be opened. Try another JPEG or PNG.');
  }
}
async function checkModel() {
  $('retry-button').disabled = true;
  $('model-status').textContent = 'Checking model…'; $('model-status').className = 'status';
  state.modelReady = false; updateButton();
  try {
    const health = await fetch('/health', { signal: AbortSignal.timeout(15000) });
    if (!health.ok) throw new Error('unavailable');
    const data = await health.json();
    if (!data.model_ready) throw new Error('missing-model');
    const response = await fetch('/classes', { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error('unavailable');
    const labels = (await response.json()).classes;
    if (!Array.isArray(labels) || !labels.length || !labels.every(label => typeof label === 'string')) throw new Error('unavailable');
    $('class-list').replaceChildren(...labels.map(label => { const chip = document.createElement('span'); chip.textContent = titleCase(label); return chip; }));
    state.modelReady = true; $('model-status').textContent = 'Model ready'; $('model-status').className = 'status ready';
    $('service-message').hidden = true;
  } catch (failure) {
    $('model-status').textContent = 'Model unavailable'; $('model-status').className = 'status offline';
    $('service-text').textContent = failure.message === 'missing-model' ? 'The model is not loaded. Add the trained weights and label file, then restart the app.' : 'The model service could not be reached. Please try again.';
    $('service-message').hidden = false; $('class-list').replaceChildren();
    const hint = document.createElement('span'); hint.textContent = 'Categories available when model is loaded'; $('class-list').append(hint);
  } finally { $('retry-button').disabled = false; updateButton(); }
}
function renderPrediction(data) {
  if (typeof data.predicted_class !== 'string' || !Number.isFinite(data.score) || data.score < 0 || data.score > 1 || !Array.isArray(data.top_predictions) || !data.top_predictions.length || !data.top_predictions.every(p => typeof p.label === 'string' && Number.isFinite(p.score) && p.score >= 0 && p.score <= 1)) throw new Error('The server returned an unexpected result. Please try again.');
  $('predicted-class').textContent = titleCase(data.predicted_class);
  $('primary-score').textContent = `${(data.score * 100).toFixed(1)}%`;
  $('prediction-list').replaceChildren(...data.top_predictions.slice(0, 3).map(prediction => {
    const row = document.createElement('li'); const labels = document.createElement('div'); labels.className = 'bar-label';
    const name = document.createElement('span'); name.textContent = titleCase(prediction.label);
    const score = document.createElement('span'); score.textContent = `${(prediction.score * 100).toFixed(1)}%`; labels.append(name, score);
    const bar = document.createElement('div'); bar.className = 'bar'; bar.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = `${prediction.score * 100}%`; bar.append(fill); row.append(labels, bar); return row;
  }));
  $('score-notice').textContent = typeof data.notice === 'string' ? data.notice : 'Model scores are not identification certainty.';
  $('result-empty').hidden = true; $('result-content').hidden = false;
  $('result-announcement').textContent = `Top model match: ${data.predicted_class}. Model score: ${(data.score * 100).toFixed(1)} percent.`;
  if (Array.from($('care-flower').options).some(option => option.value === data.predicted_class)) $('care-flower').value = data.predicted_class;
  $('care-submit').disabled = !$('care-flower').value;
}
$('identify-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!state.file || !state.modelReady || state.busy) return;
  const version = ++state.version; const controller = new AbortController(); state.controller = controller;
  const timeout = setTimeout(() => controller.abort(), 45000);
  state.busy = true; error(); resetResult(); updateButton();
  $('button-label').textContent = 'Taking a closer look…'; $('result-panel').setAttribute('aria-busy', 'true');
  $('result-announcement').textContent = 'Identifying your flower.';
  try {
    const body = new FormData(); body.append('file', state.file);
    const response = await fetch('/predict', { method: 'POST', body, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unexpected response. Please try again.'); }
    if (version !== state.version) return;
    if (!response.ok) {
      if (response.status === 503) { state.modelReady = false; $('model-status').textContent = 'Model unavailable'; $('model-status').className = 'status offline'; $('service-text').textContent = 'The model is unavailable. Check the service, then try again.'; $('service-message').hidden = false; }
      throw new Error(typeof data.detail === 'string' ? data.detail : 'The image could not be identified. Please try again.');
    }
    renderPrediction(data);
  } catch (failure) {
    if (version === state.version) { error(failure.name === 'AbortError' ? 'The request timed out. Please try again.' : (failure instanceof TypeError ? 'Connection lost. Please try again.' : failure.message)); $('result-announcement').textContent = 'Identification could not be completed.'; }
  } finally {
    clearTimeout(timeout);
    if (version === state.version) { state.busy = false; state.controller = null; $('button-label').textContent = 'Identify flower'; $('result-panel').setAttribute('aria-busy', 'false'); updateButton(); }
  }
});
$('image-file').addEventListener('change', event => selectFile(event.target.files[0]));
$('clear-button').addEventListener('click', clearSelection);
$('retry-button').addEventListener('click', checkModel);
const zone = $('drop-zone');
for (const name of ['dragenter', 'dragover']) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('dragging'); });
zone.addEventListener('drop', event => { if (event.dataTransfer.files.length !== 1) { clearSelection(); error('Choose one image at a time.'); } else selectFile(event.dataTransfer.files[0]); });
window.addEventListener('beforeunload', () => { if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); });
const careState = { version: 0, controller: null };
function resetCare(clearChoice = true) {
  careState.version++; careState.controller?.abort(); careState.controller = null;
  if (clearChoice) { $('care-flower').value = ''; $('care-question').value = ''; }
  $('care-output').hidden = true; $('care-error').hidden = true;
  $('care-submit').textContent = 'Show care guide ↗'; $('care-submit').disabled = !$('care-flower').value;
  $('care-form').setAttribute('aria-busy', 'false');
}
function renderCare(data) {
  if (!Array.isArray(data.facts) || !Array.isArray(data.sources)) throw new Error('The care service returned an unexpected response.');
  $('care-mode').textContent = data.mode === 'llm_tool_routing' ? 'LLM-assisted topic selection · reviewed sources' : 'Guided source lookup';
  $('care-reference').textContent = data.reference_plant || 'A little more information is needed';
  $('care-context').textContent = data.context_note || '';
  $('care-notice').textContent = data.notice || '';
  $('care-facts').replaceChildren(...data.facts.flatMap(fact => {
    const term = document.createElement('dt'); term.textContent = titleCase(fact.topic);
    const description = document.createElement('dd'); description.textContent = fact.text;
    const sourceIndex = data.sources.findIndex(source => source.id === fact.source_id);
    if (sourceIndex >= 0) { const citation = document.createElement('a'); citation.href = `#care-source-${sourceIndex}`; citation.textContent = ` [${sourceIndex + 1}]`; citation.setAttribute('aria-label', `Source ${sourceIndex + 1} for ${fact.topic}`); description.append(citation); }
    return [term, description];
  }));
  $('care-sources').replaceChildren(...data.sources.map((source, index) => {
    const url = new URL(source.url);
    if (url.protocol !== 'https:' || !['plants.ces.ncsu.edu', 'www.rhs.org.uk'].includes(url.hostname)) throw new Error('The care source could not be validated.');
    const row = document.createElement('p'); row.id = `care-source-${index}`;
    const link = document.createElement('a'); link.href = url.href; link.textContent = `[${index + 1}] ${source.title} ↗`; link.target = '_blank'; link.rel = 'noopener noreferrer';
    row.append(link, document.createTextNode(` · reviewed ${source.reviewed_on}`)); return row;
  }));
  $('care-limitations').textContent = data.limitations || ''; $('care-output').hidden = false;
}
$('care-flower').addEventListener('change', () => resetCare(false));
$('care-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!$('care-flower').value) return;
  resetCare(false); const version = careState.version; const controller = new AbortController(); careState.controller = controller;
  const timer = setTimeout(() => controller.abort(), 25000);
  $('care-submit').disabled = true; $('care-submit').textContent = 'Looking up sources…'; $('care-form').setAttribute('aria-busy', 'true');
  try {
    const response = await fetch('/care/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ flower: $('care-flower').value, question: $('care-question').value }) });
    if (!response.ok) throw new Error('The care guide is unavailable. Please try again.');
    const data = await response.json(); if (version !== careState.version) return;
    renderCare(data);
  } catch (failure) {
    if (version === careState.version) { $('care-error').textContent = failure.name === 'AbortError' ? 'The care request timed out. Please try again.' : 'The care guide could not be loaded. Please try again.'; $('care-error').hidden = false; }
  } finally {
    clearTimeout(timer);
    if (version === careState.version) { careState.controller = null; $('care-submit').disabled = !$('care-flower').value; $('care-submit').textContent = 'Show care guide ↗'; $('care-form').setAttribute('aria-busy', 'false'); }
  }
});
document.querySelectorAll('[data-care-question]').forEach(button => button.addEventListener('click', () => { $('care-question').value = button.dataset.careQuestion; if ($('care-flower').value) $('care-form').requestSubmit(); else { $('care-error').textContent = 'Choose a flower category first.'; $('care-error').hidden = false; } }));
checkModel();
