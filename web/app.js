const Progress = window.BTBProgress;
const { STATES, STATE_LABELS } = Progress;

let catalog = { tracks: [] };
let selectedTrackId = '';
let selectedUnitPath = '';
let selectedResourceHref = '';
let progressStore = Progress.loadProgress();
let activeUserId = progressStore.activeUserId;
let contentRequestId = 0;

const $ = (selector) => document.querySelector(selector);
const trackList = $('#track-list');
const unitList = $('#unit-list');
const detail = $('#unit-detail');
const emptyState = $('#empty-state');
const searchInput = $('#search');
const progressFilter = $('#progress-filter');
const profileSelect = $('#profile-select');
const routeSelect = $('#route-select');

init();

async function init() {
  try {
    const response = await fetch('catalog.json');
    if (!response.ok) throw new Error(`catalog load failed: ${response.status}`);
    catalog = await response.json();
  } catch (error) {
    detail.innerHTML = `<p class="empty">학습 자료 목록을 읽지 못했습니다. 저장소 루트에서 <code>python -m http.server 8000</code>를 실행한 뒤 <code>http://localhost:8000/web/</code>을 열어 주세요.</p>`;
    return;
  }

  restoreActiveView();
  renderProfiles();
  bindEvents();
  render();
}

function persistProgress() {
  try {
    Progress.saveProgress(progressStore);
  } catch (error) {
    console.warn('Progress is session-only because localStorage is unavailable.', error);
  }
}

function activeUI() {
  return Progress.userUI(progressStore, activeUserId);
}

function updateActiveUI(patch) {
  Progress.updateUserUI(progressStore, activeUserId, patch);
}

function restoreActiveView() {
  const ui = activeUI();
  if (!routeDefinitions().some((route) => route.id === ui.selectedRoute)) {
    updateActiveUI({ selectedRoute: 'full' });
  }
  const recommended = recommendedStartingUnit();
  selectedUnitPath = findUnit(ui.selectedUnit) ? ui.selectedUnit : recommended?.path || '';
  const unitTrackId = selectedUnitPath ? selectedUnitPath.split('/')[0] : '';
  const savedTrackExists = catalog.tracks.some((track) => track.id === ui.selectedTrack);
  selectedTrackId = unitTrackId || (savedTrackExists ? ui.selectedTrack : catalog.tracks[0]?.id || '');
  selectedResourceHref = '';
  updateActiveUI({ selectedTrack: selectedTrackId, selectedUnit: selectedUnitPath });
}

function currentUser() {
  return Progress.ensureUser(progressStore, activeUserId);
}

function lessonState(unitPath) {
  return Progress.lessonState(progressStore, activeUserId, unitPath);
}

function updateLesson(unitPath, patch) {
  Progress.upsertLessonProgress(progressStore, activeUserId, unitPath, patch);
  selectedUnitPath = unitPath;
  selectedTrackId = unitPath.split('/')[0];
  persistProgress();
  render();
}

function bindEvents() {
  searchInput.value = activeUI().filters?.query || '';
  progressFilter.value = activeUI().filters?.progressState || 'all';
  routeSelect.value = activeUI().selectedRoute || 'full';

  searchInput.addEventListener('input', () => {
    updateActiveUI({ filters: { query: searchInput.value } });
    persistProgress();
    renderUnits();
  });
  progressFilter.addEventListener('change', () => {
    updateActiveUI({ filters: { progressState: progressFilter.value } });
    persistProgress();
    renderUnits();
  });
  routeSelect.addEventListener('change', () => {
    updateActiveUI({ selectedRoute: routeSelect.value });
    persistProgress();
    renderOverallProgress();
    renderRouteCard();
  });
  $('#reset-filters').addEventListener('click', () => {
    searchInput.value = '';
    progressFilter.value = 'all';
    updateActiveUI({ filters: { query: '', progressState: 'all' } });
    persistProgress();
    renderUnits();
  });
  $('#resume').addEventListener('click', () => {
    const last = findLastOpenedUnit();
    if (last) selectUnit(last);
  });
  $('#add-profile').addEventListener('click', addProfile);
  $('#export-progress').addEventListener('click', exportProgress);
  $('#import-progress').addEventListener('click', () => $('#import-dialog').showModal());
  $('#confirm-import').addEventListener('click', importProgress);
  $('#reset-progress').addEventListener('click', resetProgress);
  profileSelect.addEventListener('change', () => {
    activeUserId = profileSelect.value;
    progressStore.activeUserId = activeUserId;
    restoreActiveView();
    searchInput.value = activeUI().filters?.query || '';
    progressFilter.value = activeUI().filters?.progressState || 'all';
    routeSelect.value = activeUI().selectedRoute || 'full';
    persistProgress();
    render();
  });
}

function render() {
  renderProfiles();
  renderTracks();
  renderUnits();
  renderDetail();
  renderOverallProgress();
  renderRouteCard();
}

function renderProfiles() {
  profileSelect.innerHTML = Object.entries(progressStore.users)
    .map(([id, user]) => `<option value="${escapeHtml(id)}">${escapeHtml(user.displayName || id)}</option>`)
    .join('');
  profileSelect.value = activeUserId;
  routeSelect.value = activeUI().selectedRoute || 'full';
}

function renderTracks() {
  trackList.innerHTML = catalog.tracks.map((track) => {
    const stats = trackStats(track);
    return `<button class="track-card" type="button" aria-pressed="${track.id === selectedTrackId}" data-track="${escapeHtml(track.id)}">
      <div class="track-top"><span class="track-title">${escapeHtml(track.title)}</span><span class="track-meta">${stats.done}/${stats.total}</span></div>
      <div class="track-meta">${escapeHtml(track.id)}</div>
      <p>${renderInlineSummary(track.summary || '이 트랙의 README에서 학습 방향을 확인한다.')}</p>
      <div class="progress-bar" aria-hidden="true"><span style="width:${stats.percent}%"></span></div>
    </button>`;
  }).join('');

  trackList.querySelectorAll('[data-track]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedTrackId = button.dataset.track;
      updateActiveUI({ selectedTrack: selectedTrackId });
      persistProgress();
      render();
    });
  });
}

function renderUnits() {
  const query = searchInput.value.trim().toLowerCase();
  const stateFilter = progressFilter.value;
  const selectedTrack = catalog.tracks.find((track) => track.id === selectedTrackId) || catalog.tracks[0];
  if (!selectedTrack) return;

  const units = selectedTrack.units.filter((unit) => {
    const state = lessonState(unit.path).state;
    const matchesState = stateFilter === 'all' || state === stateFilter;
    const haystack = [unit.id, unit.title, unit.objective, unit.key_terms.join(' ')].join(' ').toLowerCase();
    return matchesState && (!query || haystack.includes(query));
  });

  emptyState.hidden = units.length > 0;
  unitList.innerHTML = units.map((unit) => unitCard(unit)).join('');
  unitList.querySelectorAll('[data-unit]').forEach((button) => {
    button.addEventListener('click', () => selectUnit(button.dataset.unit));
  });
}

function unitCard(unit) {
  const progress = lessonState(unit.path);
  const outputChips = (unit.required_outputs || []).slice(0, 3).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join('');
  return `<button class="unit-card" type="button" data-unit="${escapeHtml(unit.path)}" aria-current="${unit.path === selectedUnitPath}">
    <div class="unit-top"><span class="unit-title">${escapeHtml(unit.title)}</span><span class="chip ${progress.state}">${STATE_LABELS[progress.state]}</span></div>
    <div class="unit-meta">${escapeHtml(unit.path)} · ${escapeHtml(unit.status)}</div>
    <p>${renderInlineSummary(unit.objective || '목표 설명은 사이트 안의 README 탭에서 확인하세요.')}</p>
    <div class="chips">${outputChips}</div>
  </button>`;
}

function renderDetail() {
  const unit = findUnit(selectedUnitPath);
  if (!unit) {
    detail.innerHTML = '<p class="empty">단원을 선택하면 읽을 자료, 실습 코드, 실행 결과, 메모를 한 화면에서 이어갈 수 있습니다.</p>';
    return;
  }
  const progress = lessonState(unit.path);
  const checkpoints = unit.checkpoints.length ? unit.checkpoints : ['README'];
  const checked = progress.checkpoints || {};
  const selfChecks = progress.selfChecks || {};
  const selfCheckItems = selfChecksFor(unit);
  const selfCheckStats = selfCheckProgress(selfCheckItems, selfChecks);
  const percent = completionPercent(checkpoints, checked, progress.state);
  const sections = lessonSectionsFor(unit);
  const selectedSection = sections.find((section) => hrefEquals(section.href, selectedResourceHref)) || sections[0];
  selectedResourceHref = selectedSection.href;

  detail.innerHTML = `<section class="lesson-hero">
      <div>
        <h2 id="detail-title">${escapeHtml(unit.title)}</h2>
        <p class="unit-meta">${escapeHtml(unit.path)} · ${escapeHtml(executionLabelFor(unit))} · 내 상태: ${STATE_LABELS[progress.state]}</p>
        <p>${renderInlineSummary(unit.objective || '')}</p>
        ${scopeGateFor(unit)}
      </div>
      <div>
        <div class="status-buttons" aria-label="진행 상태 변경">
          ${STATES.map((state) => `<button type="button" data-state="${state}" class="${state === progress.state ? 'active' : ''}">${STATE_LABELS[state]}</button>`).join('')}
        </div>
        <div class="progress-bar" aria-label="체크리스트 ${percent}% 완료"><span style="width:${percent}%"></span></div>
      </div>
    </section>
    <div class="lesson-workspace reader-shell">
      <aside class="lesson-guide" aria-label="학습 진행 가이드">
        <div class="start-callout">처음이라면 README와 THEORY로 목표를 잡고, 코드 실행 → 결과 해석 → 메모 순서로 진행하세요.</div>
        <h3>학습 순서</h3>
        <ol class="learning-steps">
          ${learningStepsFor(unit).map((step) => `<li><strong>${escapeHtml(step.label)}</strong><span>${escapeHtml(step.description)}</span></li>`).join('')}
        </ol>
        <h3>체크리스트</h3>
        <ul class="checklist">
          ${checkpoints.map((item) => `<li><label><input type="checkbox" data-checkpoint="${escapeHtml(item)}" ${checked[item] ? 'checked' : ''}/> ${escapeHtml(item)}</label></li>`).join('')}
        </ul>
        <h3>선행 확인</h3>
        <ul>${(unit.prereqs || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>이전 트랙과 study guide를 먼저 확인한다.</li>'}</ul>
        <h3>학습 방향</h3>
        <div class="resource-list">${studyLinksFor(unit).map((link) => `<button type="button" class="resource-button" data-resource-href="${escapeHtml(link.href)}" data-resource-label="${escapeHtml(link.label)}">${escapeHtml(link.label)}<span>${escapeHtml(link.reason)}</span></button>`).join('')}</div>
        <h3>핵심 용어</h3>
        <div class="chips">${(unit.key_terms || []).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join('') || '<span class="chip">README 참고</span>'}</div>
        <h3>남길 산출물</h3>
        <ul>${(unit.required_outputs || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>README와 analysis를 확인한다.</li>'}</ul>
        <h3>분석 질문</h3>
        <ul>${(unit.analysis_questions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>이 단원이 다음 트랙과 어떻게 연결되는지 설명한다.</li>'}</ul>
        <h3 class="self-check-heading">자가 점검 <span data-self-check-summary>${selfCheckStats.done}/${selfCheckStats.total} 완료</span></h3>
        <div class="self-check-meter" aria-label="자가 점검 ${selfCheckStats.percent}% 완료"><span style="width:${selfCheckStats.percent}%"></span></div>
        <ul class="self-checklist">
          ${selfCheckItems.map((item) => `<li><label><input type="checkbox" data-self-check="${escapeHtml(item.id)}" ${selfChecks[item.id] ? 'checked' : ''}/> ${escapeHtml(item.label)}</label><span>${escapeHtml(item.hint)}</span></li>`).join('')}
        </ul>
        <h3>내 메모</h3>
        <textarea class="notes" id="unit-note" placeholder="헷갈린 개념, 다시 볼 코드, 다음 질문을 적어 두세요. 이 브라우저에만 저장됩니다.">${escapeHtml(progress.note || '')}</textarea>
      </aside>
      <section class="lesson-reader" aria-live="polite">
        <div class="reader-header">
          <div>
            <p class="eyebrow">학습 자료</p>
            <h3>사이트 안에서 읽고 실습 흐름으로 넘어가기</h3>
          </div>
          <button id="mark-section-complete" type="button">이 자료 완료 표시</button>
        </div>
        <div class="document-tabs" role="tablist" aria-label="단원 자료">
          ${sections.map((section) => `<button type="button" role="tab" data-section-href="${escapeHtml(section.href)}" aria-selected="${hrefEquals(section.href, selectedSection.href)}">${escapeHtml(section.label)}</button>`).join('')}
        </div>
        <article id="lesson-content" class="lesson-content"><p class="empty">자료를 불러오는 중입니다.</p></article>
      </section>
    </div>`;

  bindDetailEvents(unit, checkpoints, checked, selfChecks, progress.state, selectedSection);
  loadLessonSection(unit, selectedSection);
}

function bindDetailEvents(unit, checkpoints, checked, selfChecks, currentState, selectedSection) {
  detail.querySelectorAll('[data-state]').forEach((button) => {
    button.addEventListener('click', () => updateLesson(unit.path, { state: button.dataset.state, percent: completionPercent(checkpoints, checked, button.dataset.state) }));
  });
  detail.querySelectorAll('[data-checkpoint]').forEach((checkbox) => {
    checkbox.addEventListener('change', () => {
      const next = { ...checked, [checkbox.dataset.checkpoint]: checkbox.checked };
      const nextPercent = completionPercent(checkpoints, next, currentState);
      const nextState = nextPercent === 100 ? 'done' : (currentState === 'not_started' ? 'in_progress' : currentState);
      updateLesson(unit.path, { checkpoints: next, percent: nextPercent, state: nextState });
    });
  });
  detail.querySelectorAll('[data-self-check]').forEach((checkbox) => {
    checkbox.addEventListener('change', () => {
      const next = { ...selfChecks, [checkbox.dataset.selfCheck]: checkbox.checked };
      const nextState = currentState === 'not_started' ? 'in_progress' : currentState;
      updateLesson(unit.path, { selfChecks: next, state: nextState });
    });
  });
  detail.querySelectorAll('[data-section-href]').forEach((button) => {
    button.addEventListener('click', () => {
      const section = lessonSectionsFor(unit).find((candidate) => hrefEquals(candidate.href, button.dataset.sectionHref));
      if (!section) return;
      selectedResourceHref = section.href;
      loadLessonSection(unit, section);
    });
  });
  detail.querySelectorAll('[data-resource-href]').forEach((button) => {
    button.addEventListener('click', () => {
      const section = {
        id: `resource-${button.dataset.resourceLabel}`,
        label: button.dataset.resourceLabel || '자료',
        href: button.dataset.resourceHref,
        type: 'markdown',
        checkpoint: '',
      };
      selectedResourceHref = section.href;
      loadLessonSection(unit, section);
    });
  });
  $('#unit-note').addEventListener('change', (event) => updateLesson(unit.path, { note: event.target.value }));
}

function lessonSectionsFor(unit) {
  if (unit.resources?.length) {
    return unit.resources.map((resource) => ({
      id: resource.id,
      label: resource.label,
      href: `../${resource.href}`,
      type: resource.type,
      language: resource.language,
      checkpoint: resource.checkpoint || '',
    }));
  }
  const base = `../${unit.path}`;
  return [
    { id: 'readme', label: 'README', href: `../${unit.readme}`, type: 'markdown', checkpoint: 'README' },
    { id: 'theory', label: 'THEORY', href: `${base}/THEORY.md`, type: 'markdown', checkpoint: 'THEORY' },
    { id: 'prereqs', label: 'PREREQS', href: `${base}/PREREQS.md`, type: 'markdown', checkpoint: 'PREREQS' },
    { id: 'scratch', label: 'scratch_lab.py', href: `${base}/scratch_lab.py`, type: 'code', language: 'python', checkpoint: 'scratch lab' },
    { id: 'framework', label: 'framework_lab.py', href: `${base}/framework_lab.py`, type: 'code', language: 'python', checkpoint: 'framework lab' },
    { id: 'analysis-code', label: 'analysis.py', href: `${base}/analysis.py`, type: 'code', language: 'python', checkpoint: 'analysis script' },
    { id: 'analysis-note', label: 'analysis.md', href: `${base}/analysis.md`, type: 'markdown', checkpoint: 'analysis note' },
    { id: 'reflection', label: 'reflection.md', href: `${base}/reflection.md`, type: 'markdown', checkpoint: 'reflection' },
  ];
}

async function loadLessonSection(unit, section) {
  const requestId = ++contentRequestId;
  const content = $('#lesson-content');
  const markButton = $('#mark-section-complete');
  if (!content) return;

  selectedResourceHref = section.href;
  detail.querySelectorAll('[data-section-href]').forEach((button) => {
    const selected = hrefEquals(button.dataset.sectionHref, section.href);
    button.setAttribute('aria-selected', String(selected));
  });

  content.innerHTML = `<p class="empty">${escapeHtml(section.label)} 자료를 사이트 안으로 불러오는 중입니다.</p>`;
  updateMarkSectionButton(unit, section, markButton);
  try {
    const text = await fetchLessonDocument(section.href);
    if (requestId !== contentRequestId) return;
    if (section.type === 'code') {
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(section.label)}</span><code>${escapeHtml(cleanHref(section.href))}</code></div>${renderCodeExplanation(section, text)}${renderRunPanel(section, unit)}<pre class="code-block"><code>${escapeHtml(annotateCodeWithInlineHints(section, text))}</code></pre>`;
      bindRunButton(section, unit);
    } else {
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(section.label)}</span><code>${escapeHtml(cleanHref(section.href))}</code></div>${renderMarkdown(text, section.href)}`;
      bindInlineDocLinks(unit, section.href);
    }
  } catch (error) {
    if (requestId !== contentRequestId) return;
    content.innerHTML = `<p class="empty">${escapeHtml(section.label)}을 사이트 안에서 불러오지 못했습니다. 저장소 루트에서 <code>python -m http.server 8000</code>을 실행했는지 확인하세요.<br><code>${escapeHtml(cleanHref(section.href))}</code></p>`;
  }
}

async function fetchLessonDocument(href) {
  const response = await fetch(href, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`document load failed: ${response.status}`);
  return response.text();
}

function renderRunPanel(section, unit) {
  if (!isRunnableCodeSection(section)) return '';
  const plan = runPlanFor(section, unit);
  return `<section class="run-panel" aria-label="Python 코드 실행">
    <div>
      <p class="eyebrow">실행 결과</p>
      <h4>코드를 실행해 관측값 확인하기</h4>
      <p>클릭하면 이 파일을 실행하고, 아래에 종료 코드·선택된 장치·출력을 정리해 보여줍니다. 실행 서버는 idle GPU를 찾고 없으면 CPU로 내려갑니다.</p>
    </div>
    <div class="run-actions">
      <button type="button" data-run-code data-run-path="${escapeHtml(cleanHref(section.href))}">${escapeHtml(section.label)} 실행</button>
      <span class="run-status" data-run-status>아직 실행 전입니다.</span>
    </div>
    <div class="run-primer" aria-label="실행 전 확인">
      <strong>실행 전에 볼 것</strong>
      <dl>
        <div><dt>예상 산출물</dt><dd>${escapeHtml(plan.artifacts.join(', '))}</dd></div>
        <div><dt>봐야 할 숫자</dt><dd>${escapeHtml(plan.metrics.join(', '))}</dd></div>
        <div><dt>좋은 결과 기준</dt><dd>${escapeHtml(plan.goodOutcome)}</dd></div>
      </dl>
    </div>
    <div class="run-insights" data-run-insights hidden></div>
    <pre class="run-output" data-run-output hidden></pre>
  </section>`;
}

function isRunnableCodeSection(section) {
  return section.type === 'code' && /(?:scratch_lab|framework_lab|analysis|run_stage)\.py$/.test(cleanHref(section.href));
}

function bindRunButton(section, unit) {
  const button = $('#lesson-content [data-run-code]');
  if (!button) return;
  button.addEventListener('click', () => runPythonSection(section, button, unit));
}

async function runPythonSection(section, button, unit) {
  const panel = button.closest('.run-panel');
  const output = panel?.querySelector('[data-run-output]');
  const status = panel?.querySelector('[data-run-status]');
  const insights = panel?.querySelector('[data-run-insights]');
  if (!output || !status) return;

  button.disabled = true;
  output.hidden = false;
  if (insights) {
    insights.hidden = true;
    insights.innerHTML = '';
  }
  output.textContent = '실행 중입니다...';
  status.textContent = '실행 중';
  try {
    const response = await fetch('/api/run-python', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: cleanHref(section.href) }),
    });
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
      ? await response.json()
      : { error: await response.text(), status: response.status };
    if (!response.ok && response.status === 501) {
      output.textContent = staticServerHelp('501 Unsupported method: 현재 서버가 POST /api/run-python을 지원하지 않는 정적 서버입니다.');
      status.textContent = '정적 서버로 실행 중';
      return;
    }
    if (!response.ok && !('returncode' in payload)) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    output.textContent = formatRunResult(payload);
    if (insights) {
      insights.innerHTML = renderRunInsights(payload, section, unit);
      insights.hidden = false;
    }
    status.textContent = payload.returncode === 0 ? '실행 완료' : `종료 코드 ${payload.returncode}`;
  } catch (error) {
    output.textContent = staticServerHelp(error.message);
    status.textContent = '실행 서버 필요';
  } finally {
    button.disabled = false;
  }
}

function formatRunResult(payload) {
  const command = Array.isArray(payload.command) ? payload.command.join(' ') : `python ${payload.path || ''}`.trim();
  const lines = [
    `명령: ${command}`,
    `종료 코드: ${payload.returncode}`,
  ];
  if (payload.runner) lines.push(`실행 환경: ${formatRunnerSummary(payload.runner)}`);
  if (payload.duration_seconds !== undefined) lines.push(`실행 시간: ${payload.duration_seconds}s`);
  if (payload.stdout) lines.push('', '[표준 출력]', payload.stdout.trimEnd());
  if (payload.stderr) lines.push('', '[오류 출력]', payload.stderr.trimEnd());
  if (!payload.stdout && !payload.stderr) lines.push('', '(출력 없음)');
  return lines.join('\n');
}

function formatRunnerSummary(runner) {
  const gpu = runner.gpu_index !== undefined && runner.gpu_index !== null ? `, gpu=${runner.gpu_index}` : '';
  return `${runner.environment || 'current python'}, device=${runner.device || 'unknown'}${gpu} (${runner.device_reason || 'runner selected'})`;
}

function runPlanFor(section, unit) {
  return {
    artifacts: expectedArtifactsForRun(section, unit),
    metrics: importantNumbersForRun(section, unit),
    goodOutcome: goodOutcomeForRun(section, unit),
  };
}

function expectedArtifactsForRun(section, unit) {
  const path = cleanHref(section.href);
  const declared = (unit?.required_outputs || []).filter((item) => !/^runnable README|theory note|prerequisite checklist$/i.test(item));
  if (path.endsWith('run_stage.py')) return declared.length ? declared : ['artifacts/<timestamp>/metrics.json', 'figures/', 'predictions/', 'summary.md'];
  if (path.endsWith('analysis.py')) return ['analysis markdown 또는 latest_report.md', 'observed metrics json', ...declared.filter((item) => /analysis|report|observed/i.test(item))].slice(0, 4);
  if (path.endsWith('framework_lab.py')) return declared.filter((item) => /framework|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['framework 실행 요약']).slice(0, 4);
  if (path.endsWith('scratch_lab.py')) return declared.filter((item) => /scratch|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['scratch 실행 요약']).slice(0, 4);
  return declared.length ? declared.slice(0, 4) : ['metrics json', 'figure 또는 markdown report'];
}

function importantNumbersForRun(section, unit) {
  const path = cleanHref(section.href);
  const terms = unit?.key_terms || [];
  if (path.endsWith('run_stage.py')) return ['primary metric', 'baseline 대비 best model', 'train/eval split 또는 sample count'];
  if (path.endsWith('analysis.py')) return ['missing artifact 수', '실패 사례 수', 'analysis가 강조한 핵심 metric'];
  if (path.endsWith('framework_lab.py')) return ['loss 또는 accuracy 추세', 'scratch와 같은 shape/metric인지', 'device와 seed'];
  if (path.endsWith('scratch_lab.py')) return ['입력/출력 shape', '핵심 계산 결과', terms[0] ? `${terms[0]} 관측값` : '작은 toy metric'];
  return ['return code', 'metric', 'artifact path'];
}

function goodOutcomeForRun(section, unit) {
  const path = cleanHref(section.href);
  const deterministic = unit?.deterministic ? ' 같은 seed로 재실행해도 핵심 숫자가 유지되어야 합니다.' : '';
  if (path.endsWith('run_stage.py')) return `종료 코드 0, metrics/figure/prediction artifact가 생기고 README의 baseline 질문에 답할 수 있으면 좋습니다.${deterministic}`;
  if (path.endsWith('analysis.py')) return `이전 실행 산출물을 빠짐없이 읽고 analysis markdown에 실패 사례와 다음 실험 질문이 남으면 좋습니다.${deterministic}`;
  if (path.endsWith('framework_lab.py')) return `framework 결과가 scratch 기준선과 설명 가능한 차이만 보이고, device/seed가 출력에 남으면 좋습니다.${deterministic}`;
  if (path.endsWith('scratch_lab.py')) return `작은 입력에서 shape와 계산값을 직접 설명할 수 있고, metrics json/그림이 analysis 기준선으로 남으면 좋습니다.${deterministic}`;
  return `종료 코드 0과 재확인 가능한 artifact path가 남으면 좋습니다.${deterministic}`;
}

function renderRunInsights(payload, section, unit) {
  const highlights = extractMetricHighlights(payload);
  const ok = Number(payload.returncode) === 0;
  const runner = payload.runner || {};
  const plan = runPlanFor(section, unit);
  const artifactHint = artifactHintForRun(section, payload, unit);
  const nextQuestions = runFollowupQuestions(section, payload, highlights, unit);
  return `<section aria-label="실행 관찰 카드">
    <p class="eyebrow">실행 관찰 카드</p>
    <div class="insight-grid">
      <div><strong>상태</strong><span>${ok ? '정상 실행' : `확인 필요 · 종료 코드 ${escapeHtml(payload.returncode)}`}</span></div>
      <div><strong>실행 환경</strong><span>${escapeHtml(runner.device || 'unknown')}${runner.gpu_index !== undefined && runner.gpu_index !== null ? ` · GPU ${escapeHtml(runner.gpu_index)}` : ''}</span></div>
      <div><strong>산출물</strong><span>${escapeHtml(artifactHint)}</span></div>
    </div>
    <h5>예상 산출물</h5>
    <ul>${plan.artifacts.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    <h5>봐야 할 숫자</h5>
    <ul>${highlights.length ? highlights.map((item) => `<li><code>${escapeHtml(item.path)}</code>: ${escapeHtml(item.value)}</li>`).join('') : plan.metrics.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    <h5>좋은 결과 기준</h5>
    <p>${escapeHtml(plan.goodOutcome)}</p>
    <h5>다음 질문</h5>
    <ul>${nextQuestions.map((question) => `<li>${escapeHtml(question)}</li>`).join('')}</ul>
  </section>`;
}

function extractMetricHighlights(payload) {
  const parsed = parseJsonFromStdout(payload.stdout || '');
  if (!parsed) return [];
  const candidates = [];
  const visit = (value, path) => {
    if (candidates.length >= 8) return;
    if (typeof value === 'number' || typeof value === 'boolean') {
      candidates.push({ path, value: String(value) });
      return;
    }
    if (typeof value === 'string' && value.length <= 80 && /shape|device|path|file|artifact|metric|loss|accuracy|f1|rmse|score/i.test(path)) {
      candidates.push({ path, value });
      return;
    }
    if (Array.isArray(value)) {
      if (value.length && value.length <= 8 && value.every((item) => typeof item === 'number' || typeof item === 'string')) {
        candidates.push({ path, value: `[${value.join(', ')}]` });
      }
      return;
    }
    if (value && typeof value === 'object') {
      for (const [key, child] of Object.entries(value)) {
        const nextPath = path ? `${path}.${key}` : key;
        if (/loss|accuracy|f1|rmse|mae|score|shape|count|rate|device|artifact|saved|path|metric/i.test(nextPath)) {
          visit(child, nextPath);
        } else if (typeof child === 'object' && child !== null && !Array.isArray(child)) {
          visit(child, nextPath);
        }
        if (candidates.length >= 8) break;
      }
    }
  };
  visit(parsed, '');
  return candidates.slice(0, 5);
}

function parseJsonFromStdout(stdout) {
  const text = String(stdout || '').trim();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch (_) {
        return null;
      }
    }
  }
  return null;
}

function artifactHintForRun(section, payload, unit) {
  const path = cleanHref(section.href);
  if (path.endsWith('analysis.py')) return 'analysis.md 또는 observed metrics가 갱신됐는지 확인하세요.';
  if (path.endsWith('run_stage.py')) return 'stage artifacts 아래 metrics, figures, predictions, summary를 확인하세요.';
  if (path.endsWith('framework_lab.py')) return 'framework metrics와 figure를 scratch 결과와 나란히 비교하세요.';
  if (path.endsWith('scratch_lab.py')) return 'scratch metrics json과 작은 표/그림이 analysis의 기준선입니다.';
  const expected = expectedArtifactsForRun(section, unit)[0];
  if (payload.path) return `${payload.path} 실행 결과와 ${expected}를 확인하세요.`;
  return `실행 결과와 ${expected}를 확인하세요.`;
}

function runFollowupQuestions(section, payload, highlights, unit) {
  const path = cleanHref(section.href);
  const questions = [];
  if (Number(payload.returncode) !== 0) {
    questions.push('오류 출력에서 missing file, dependency, timeout 중 무엇이 원인인지 분류하세요.');
    questions.push('CPU/GPU/conda 환경을 바꿔 재실행해야 하는지 확인하세요.');
    return questions;
  }
  if (highlights.length) {
    questions.push('가장 중요한 숫자 하나를 README의 성공 기준이나 analysis 질문과 연결해 설명해 보세요.');
  } else {
    questions.push('출력 원문에서 shape, loss, accuracy, 저장 경로 중 무엇을 확인해야 하는지 표시해 보세요.');
  }
  if (unit?.analysis_questions?.[0]) questions.push(`분석 질문과 연결: ${unit.analysis_questions[0]}`);
  if (path.endsWith('scratch_lab.py')) questions.push('scratch 결과가 framework 결과와 같아야 하는 부분과 달라도 되는 부분을 구분하세요.');
  else if (path.endsWith('framework_lab.py')) questions.push('framework가 자동으로 처리한 부분이 scratch 코드의 어느 줄과 대응되는지 찾아보세요.');
  else if (path.endsWith('analysis.py')) questions.push('analysis가 말하는 실패 사례나 다음 실험 질문을 내 메모에 한 줄로 남기세요.');
  else if (path.endsWith('run_stage.py')) questions.push('dataset.py와 experiment.py 중 어떤 단계가 이 숫자에 가장 크게 영향을 줬는지 추적하세요.');
  questions.push('다음 단원으로 가기 전에 자가 점검을 체크할 수 있는지 확인하세요.');
  return questions.slice(0, 4);
}

function staticServerHelp(detail) {
  const safeDetail = formatStaticServerDetail(detail);
  return [
    '지금은 읽기 전용 서버로 열려 있어 코드 실행 버튼을 사용할 수 없습니다.',
    '터미널에서 서버를 Ctrl+C로 멈춘 뒤, 저장소 루트에서 아래 명령으로 다시 열어 주세요.',
    '',
    currentStudyServerCommand(),
    `${window.location.origin}/web/`,
    '',
    'conda 환경을 쓰려면 예:',
    `${currentStudyServerCommand()} --conda-env btb`,
    'GPU를 쓰지 않으려면 --device cpu, 특정 GPU를 쓰려면 --device cuda --gpu-index 0을 붙입니다.',
    '',
    `상세: ${safeDetail}`,
  ].join('\n');
}

function formatStaticServerDetail(detail) {
  const text = String(detail || '').trim();
  if (!text) return '501 Unsupported method: 정적 서버는 Python 실행 API를 제공하지 않습니다.';
  if (text.includes('<!DOCTYPE') || text.includes('<html')) {
    return '501 Unsupported method: 정적 서버가 실행 요청을 HTML 오류 페이지로 응답했습니다.';
  }
  if (text.includes('Unsupported method') || text.includes('501')) {
    return '501 Unsupported method: 정적 서버는 Python 실행 API를 제공하지 않습니다.';
  }
  return text.slice(0, 240);
}

function currentStudyServerCommand() {
  const port = window.location.port || '8000';
  return `python scripts/study_server.py --port ${port} --device auto`;
}

function renderCodeExplanation(section, source) {
  const explanation = codeExplanationFor(section, source);
  return `<section class="code-explanation" aria-label="코드 읽기 안내">
    <div>
      <p class="eyebrow">코드 읽기 안내</p>
      <h4>${escapeHtml(explanation.title)}</h4>
      <p>${escapeHtml(explanation.summary)}</p>
    </div>
    <dl>
      <div><dt>이 파일은 무엇인가</dt><dd>${escapeHtml(explanation.what)}</dd></div>
      <div><dt>어떻게 읽으면 좋은가</dt><dd>${escapeHtml(explanation.howToRead)}</dd></div>
      <div><dt>실행하면 남는 결과</dt><dd>${escapeHtml(explanation.outputs)}</dd></div>
      <div><dt>읽어볼 함수</dt><dd>${escapeHtml(explanation.functions.join(', ') || '상단 설정값과 저장 흐름')}</dd></div>
    </dl>
  </section>`;
}

function annotateCodeWithInlineHints(section, source) {
  return annotateFunctionRoleHints(annotateArtifactLocations(source, section), section);
}

function annotateArtifactLocations(source, section) {
  const artifactHint = '# 학습 포인트: 이 경로가 실행 후 metrics/figure/report가 남는 위치입니다.';
  const reportHint = '# 학습 포인트: analysis.py가 최종 해석 문서를 쓰는 위치입니다.';
  let annotated = source;
  if (/^ARTIFACT_DIR\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^ARTIFACT_DIR\s*=)/m, `${artifactHint}\n$1`);
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^REPORT\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^REPORT\s*=)/m, `${reportHint}\n$1`);
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^SCRATCH\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^SCRATCH\s*=)/m, '# 학습 포인트: scratch/framework metrics를 analysis 입력으로 다시 읽습니다.\n$1');
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^ANALYSIS_PATH\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^ANALYSIS_PATH\s*=)/m, '# 학습 포인트: 분석 결과 markdown이 저장되는 위치입니다.\n$1');
  }
  if (/^with\s+torch\.no_grad\(\):/m.test(annotated)) {
    annotated = annotated.replace(/(^with\s+torch\.no_grad\(\):)/m, '# 학습 포인트: 여기부터는 학습이 아니라 평가/추론 구간입니다.\n$1');
  }
  return annotated;
}

function annotateFunctionRoleHints(source, section) {
  return functionRoleHintsFor(section, source).reduce((annotated, { name, hint }) => {
    const pattern = new RegExp(`(^def\\s+${escapeRegExp(name)}\\s*\\()`, 'm');
    if (!pattern.test(annotated)) return annotated;
    return annotated.replace(pattern, `# 코드 읽기 힌트: ${hint}\n$1`);
  }, source);
}

function functionRoleHintsFor(section, source) {
  const names = extractPythonSymbols(source).map((symbol) => symbol.replace(/\(\)$/, ''));
  const seen = new Set();
  const hints = [];
  for (const name of names) {
    const hint = roleHintForFunction(name, section);
    if (!hint || seen.has(name)) continue;
    seen.add(name);
    hints.push({ name, hint });
    if (hints.length >= 5) break;
  }
  return hints;
}

function roleHintForFunction(name, section) {
  const normalized = name.toLowerCase();
  if (normalized === 'run' || normalized === 'main') return sectionSpecificRunHint(section);
  if (normalized.includes('forward')) return 'tensor 입력이 logit·embedding·action 같은 모델 출력으로 바뀌는 계산 경로입니다.';
  if (normalized.includes('train')) return 'batch → loss → optimizer step이 연결되는 학습 루프입니다.';
  if (normalized.includes('evaluate') || normalized.includes('metric') || normalized.includes('score')) return '단원에서 비교할 지표를 계산하므로 README의 성공 기준과 나란히 확인하세요.';
  if (normalized.includes('compute') || normalized.includes('calculate')) return '중간 텐서나 수치를 최종 metrics로 바꾸는 계산입니다.';
  if (normalized.includes('build') || normalized.includes('create') || normalized.includes('prepare') || normalized.includes('make')) return 'toy data, model, config 중 무엇을 고정해 실험 조건을 만드는지 확인하세요.';
  if (normalized.includes('generate') || normalized.includes('sample') || normalized.includes('decode')) return '모델 출력이 사람이 읽을 수 있는 token·caption·action으로 바뀌는 지점입니다.';
  if (normalized.includes('write') || normalized.includes('save')) return '브라우저와 analysis.py가 다시 읽을 artifact를 파일로 남기는 지점입니다.';
  if (normalized.includes('load') || normalized.includes('read')) return '이전 실행 산출물을 다시 읽어 분석 입력으로 바꾸는 지점입니다.';
  return '';
}

function sectionSpecificRunHint(section) {
  const path = cleanHref(section.href);
  if (path.endsWith('scratch_lab.py')) return '작은 입력 예제를 만들고 직접 계산한 뒤 metrics/artifacts로 남기는 흐름입니다.';
  if (path.endsWith('framework_lab.py')) return '프레임워크 모델·학습/평가 설정을 묶어 scratch 결과와 비교할 metrics를 만듭니다.';
  if (path.endsWith('analysis.py')) return 'scratch/framework 산출물을 읽고 빠진 결과를 확인한 뒤 analysis.md 또는 요약을 작성합니다.';
  if (path.endsWith('run_stage.py')) return 'CLI 인자를 받아 실제 stage 실행 함수로 넘기는 연결부입니다.';
  return '';
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function codeExplanationFor(section, source) {
  const path = cleanHref(section.href);
  const functions = extractPythonSymbols(source);
  if (path.endsWith('scratch_lab.py')) {
    return {
      title: 'scratch_lab.py는 개념을 직접 계산해 보는 코드입니다.',
      summary: '라이브러리 편의 기능에 기대기 전에, 작은 숫자와 단계별 계산으로 이 단원의 핵심 원리를 확인합니다.',
      what: '이 파일은 모델이나 metric을 아주 작은 예제로 직접 구성해, README/THEORY의 설명이 실제 숫자로 어떻게 바뀌는지 보여주는 scratch 실습입니다.',
      howToRead: '위쪽의 데이터/설정 → 중간의 계산 함수 → 아래쪽의 artifact 저장 순서로 읽으면 됩니다. 먼저 입력 shape와 중간 변수 이름을 보고, 마지막에 저장되는 metrics를 확인하세요.',
      outputs: '보통 artifacts 아래 scratch metrics json, 작은 svg/표, 또는 실행 요약이 남습니다. 이 결과는 analysis.py가 비교·해석하는 기준선입니다.',
      functions,
    };
  }
  if (path.endsWith('framework_lab.py')) {
    return {
      title: 'framework_lab.py는 같은 아이디어를 프레임워크로 확인하는 코드입니다.',
      summary: 'scratch에서 본 계산을 PyTorch/sklearn 같은 도구로 다시 실행해, 실제 연구·개발 코드의 구조와 결과를 비교합니다.',
      what: '이 파일은 단원의 핵심 개념을 프레임워크 API로 구현해 재현성, 학습 루프, metric 계산, 저장 포맷을 확인하는 실습입니다.',
      howToRead: '데이터 준비 → 모델/파이프라인 정의 → 학습 또는 추론 → metrics/artifacts 저장 순서로 따라가세요. scratch와 같은 이름의 지표가 어떻게 대응되는지 비교하면 좋습니다.',
      outputs: '보통 framework metrics json, 결과 figure, predictions 샘플이 남습니다. scratch 결과와 나란히 보며 프레임워크가 자동으로 처리한 부분을 찾습니다.',
      functions,
    };
  }
  if (path.endsWith('analysis.py')) {
    return {
      title: 'analysis.py는 실행 결과를 공부 가능한 해석으로 바꾸는 코드입니다.',
      summary: 'scratch/framework가 만든 metrics와 artifacts를 읽고, 무엇이 잘 됐고 어디서 실패했는지 한국어 분석 노트로 정리합니다.',
      what: '이 파일은 실험 산출물을 검증하고, 핵심 수치·실패 사례·다음 질문을 analysis.md나 summary 형태로 정리하는 분석 스크립트입니다.',
      howToRead: '입력 파일을 읽는 부분 → metric 검증/집계 → markdown 문장 생성 → 저장 경로 순서로 읽으세요. 예외 메시지는 어떤 산출물이 빠졌는지 알려주는 체크리스트 역할을 합니다.',
      outputs: 'analysis.md, summary.md, observed metrics json 같은 해석 산출물이 남습니다. 단원을 완료할 때는 이 파일의 질문에 답할 수 있어야 합니다.',
      functions,
    };
  }
  if (path.endsWith('dataset.py')) {
    return {
      title: 'dataset.py는 ML stage의 입력 표를 만드는 코드입니다.',
      summary: '원본 데이터를 읽고, feature matrix X와 label y가 어떤 기준으로 나뉘는지 확인하는 출발점입니다.',
      what: '이 파일은 stage 전용 데이터 로딩, train/test split, feature/label 구성, 결측·범주형 처리 준비를 담당합니다.',
      howToRead: '데이터를 불러오는 함수 → feature column 선택 → target 생성 → split/전처리 입력 형태 순서로 읽으세요. 마지막에 experiment.py가 기대하는 반환 shape를 확인합니다.',
      outputs: '대개 직접 artifact를 저장하기보다, experiment.py가 학습과 평가에 사용할 X/y 또는 dataset bundle을 넘깁니다.',
      functions,
    };
  }
  if (path.endsWith('experiment.py')) {
    return {
      title: 'experiment.py는 ML stage의 실제 실험 흐름입니다.',
      summary: 'baseline, 전처리, 모델 학습, metric 계산, artifact 저장이 한곳에서 연결되므로 이 stage의 핵심 로직입니다.',
      what: '이 파일은 dataset.py가 만든 입력을 받아 모델을 학습·비교하고, metrics/figures/predictions를 저장하는 orchestration 코드입니다.',
      howToRead: '설정값 → 데이터 준비 호출 → baseline/model 정의 → fit/predict → metric 저장 순서로 읽으세요. run_stage.py는 보통 이 흐름을 CLI에서 호출만 합니다.',
      outputs: 'artifacts 아래 config, metrics, prediction sample, figure가 남고 report.py나 analysis.py가 이를 읽어 해석합니다.',
      functions,
    };
  }
  if (path.endsWith('run_stage.py')) {
    return {
      title: 'run_stage.py는 ML stage를 한 번에 실행하는 진입 코드입니다.',
      summary: 'dataset.py와 experiment.py에 흩어진 준비·학습·평가 흐름을 CLI에서 재현 가능하게 묶습니다.',
      what: '이 파일은 stage 전용 실험을 실행하고, 선택된 CPU/GPU 환경에서 metrics와 figures를 artifacts 폴더에 남기도록 연결하는 작은 실행 스크립트입니다.',
      howToRead: '인자 파싱 → seed/device 설정 → experiment.run_stage 호출 → JSON 요약 출력 순서로 읽으세요. 실제 모델 비교와 저장 로직은 experiment.py에서 이어서 확인합니다.',
      outputs: 'artifacts 아래 metrics, config, predictions, figures와 터미널 JSON 요약이 남습니다. 실패하면 실행 환경이나 의존성 확인이 필요한 지점입니다.',
      functions,
    };
  }
  return {
    title: `${section.label} 코드 설명`,
    summary: '이 코드는 단원 실습을 재현 가능하게 실행하기 위한 보조 코드입니다.',
    what: '파일 이름과 README의 실행 순서를 함께 보며 역할을 확인하세요.',
    howToRead: '상단 설정, 데이터 준비, 계산/호출 지점, 저장 로직 순서로 읽으면 됩니다.',
    outputs: '실행 요약, metrics json, markdown, figure 중 하나 이상의 산출물을 남깁니다.',
    functions,
  };
}

function extractPythonSymbols(source) {
  return Array.from(source.matchAll(/^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/gm))
    .map((match) => `${match[1]}()`)
    .slice(0, 8);
}

function updateMarkSectionButton(unit, section, button) {
  if (!button) return;
  if (!section.checkpoint) {
    button.disabled = true;
    button.textContent = '참고 자료';
    return;
  }
  const checked = lessonState(unit.path).checkpoints || {};
  button.disabled = Boolean(checked[section.checkpoint]);
  button.textContent = checked[section.checkpoint] ? '완료 표시됨' : `${section.checkpoint} 완료 표시`;
  button.onclick = () => markSectionComplete(unit, section);
}

function markSectionComplete(unit, section) {
  if (!section.checkpoint) return;
  const progress = lessonState(unit.path);
  const checkpoints = unit.checkpoints.length ? unit.checkpoints : ['README'];
  const next = { ...(progress.checkpoints || {}), [section.checkpoint]: true };
  const nextPercent = completionPercent(checkpoints, next, progress.state);
  const nextState = nextPercent === 100 ? 'done' : (progress.state === 'not_started' ? 'in_progress' : progress.state);
  updateLesson(unit.path, { checkpoints: next, percent: nextPercent, state: nextState });
}

function bindInlineDocLinks(unit, baseHref) {
  $('#lesson-content').querySelectorAll('[data-doc-href]').forEach((button) => {
    button.addEventListener('click', () => {
      const section = {
        id: `inline-${button.dataset.docLabel}`,
        label: button.dataset.docLabel || '문서',
        href: button.dataset.docHref,
        type: 'markdown',
        checkpoint: '',
      };
      selectedResourceHref = section.href;
      loadLessonSection(unit, section);
    });
  });
}

function renderMarkdown(markdown, baseHref) {
  const lines = markdown.replaceAll('\r\n', '\n').split('\n');
  const html = [];
  let listType = '';
  let inCode = false;
  let codeLines = [];
  let tableRows = [];

  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = '';
    }
  };
  const flushCode = () => {
    html.push(`<pre class="code-block"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    codeLines = [];
  };
  const flushTable = () => {
    if (!tableRows.length) return;
    const rows = tableRows
      .map((line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim()))
      .filter((cells) => !cells.every((cell) => /^:?-{3,}:?$/.test(cell)));
    if (rows.length) {
      const [head, ...body] = rows;
      html.push('<div class="table-wrap"><table>');
      html.push(`<thead><tr>${head.map((cell) => `<th>${inlineMarkdown(cell, baseHref)}</th>`).join('')}</tr></thead>`);
      if (body.length) html.push(`<tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell, baseHref)}</td>`).join('')}</tr>`).join('')}</tbody>`);
      html.push('</table></div>');
    }
    tableRows = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) {
      flushTable();
      closeList();
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        inCode = true;
        codeLines = [];
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (!trimmed) {
      flushTable();
      closeList();
      continue;
    }
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      closeList();
      tableRows.push(line);
      continue;
    }
    flushTable();

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${inlineMarkdown(heading[2], baseHref)}</h${level}>`);
      continue;
    }
    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    if (unordered) {
      if (listType !== 'ul') {
        closeList();
        html.push('<ul>');
        listType = 'ul';
      }
      html.push(`<li>${inlineMarkdown(unordered[1], baseHref)}</li>`);
      continue;
    }
    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      if (listType !== 'ol') {
        closeList();
        html.push('<ol>');
        listType = 'ol';
      }
      html.push(`<li>${inlineMarkdown(ordered[1], baseHref)}</li>`);
      continue;
    }
    if (trimmed.startsWith('>')) {
      closeList();
      html.push(`<blockquote>${inlineMarkdown(trimmed.replace(/^>\s?/, ''), baseHref)}</blockquote>`);
      continue;
    }
    closeList();
    html.push(`<p>${inlineMarkdown(trimmed, baseHref)}</p>`);
  }
  if (inCode) flushCode();
  flushTable();
  closeList();
  return html.join('\n');
}

function inlineMarkdown(text, baseHref) {
  let escaped = escapeHtml(text);
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
  escaped = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    const clean = href.replace(/&amp;/g, '&');
    if (isLocalMarkdownHref(clean)) {
      const resolved = resolveDocHref(clean, baseHref);
      return `<button type="button" class="inline-doc-link" data-doc-href="${escapeHtml(resolved)}" data-doc-label="${escapeHtml(label)}">${label}</button>`;
    }
    return `<a href="${escapeHtml(clean)}" rel="noreferrer">${label}</a>`;
  });
  return escaped;
}

function renderInlineSummary(text) {
  let escaped = escapeHtml(stripMarkdownLinks(text));
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
  return escaped;
}

function stripMarkdownLinks(text) {
  return String(text ?? '').replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1');
}

function isLocalMarkdownHref(href) {
  return !/^[a-z]+:/i.test(href) && !href.startsWith('#') && href.split('#')[0].endsWith('.md');
}

function resolveDocHref(href, baseHref) {
  try {
    return new URL(href, new URL(baseHref, window.location.href)).href;
  } catch (_) {
    return href;
  }
}

function cleanHref(href) {
  try {
    const url = new URL(href, window.location.href);
    return url.pathname.replace(/^\//, '');
  } catch (_) {
    return href;
  }
}

function hrefEquals(left, right) {
  if (!left || !right) return false;
  try {
    return new URL(left, window.location.href).href === new URL(right, window.location.href).href;
  } catch (_) {
    return left === right;
  }
}

function recommendedStartingUnit() {
  const units = routeUnits(activeUI().selectedRoute || 'full');
  return units.find((unit) => lessonState(unit.path).state !== 'done') || units[0] || null;
}

function routeDefinitions() {
  return [
    {
      id: 'full',
      label: '전체 1-pass',
      description: '00부터 10까지 모든 트랙을 한 번씩 지나가는 가장 촘촘한 경로입니다.',
      include: () => true,
    },
    {
      id: 'llm',
      label: 'LLM/RLHF 빠른 경로',
      description: '기초·ML·DL·NLP를 거쳐 LLM pretraining, SFT, preference/RLHF까지 우선 도달합니다.',
      include: (unit) => ['00_foundations', '01_ml', '02_deep_learning', '03_nlp_bridge', '04_nlp', '05_advanced_nlp_llm'].includes(unit.path.split('/')[0]),
    },
    {
      id: 'multimodal',
      label: 'Multimodal/VLA 경로',
      description: 'LLM 기반을 만든 뒤 multimodal bridge, applied multimodal, VLA action-token 입구로 이어집니다.',
      include: (unit) => ['00_foundations', '01_ml', '02_deep_learning', '03_nlp_bridge', '04_nlp', '05_advanced_nlp_llm', '08_multimodal_bridge', '09_multimodal', '10_vla'].includes(unit.path.split('/')[0]),
    },
    {
      id: 'systems',
      label: 'Systems 심화 경로',
      description: '모델 학습 감각 이후 distributed/system, frontier lab 실험 운영 능력을 강화합니다.',
      include: (unit) => ['00_foundations', '02_deep_learning', '06_training_systems', '07_frontier_labs'].includes(unit.path.split('/')[0]),
    },
  ];
}

function selectedRouteDefinition() {
  const selected = activeUI().selectedRoute || 'full';
  return routeDefinitions().find((route) => route.id === selected) || routeDefinitions()[0];
}

function routeUnits(routeId) {
  const route = routeDefinitions().find((candidate) => candidate.id === routeId) || routeDefinitions()[0];
  return catalog.tracks.flatMap((track) => track.units).filter((unit) => route.include(unit));
}

function routeProgress(routeId) {
  const units = routeUnits(routeId);
  const done = units.filter((unit) => lessonState(unit.path).state === 'done').length;
  const inProgress = units.filter((unit) => lessonState(unit.path).state === 'in_progress').length;
  return {
    total: units.length,
    done,
    inProgress,
    percent: units.length ? Math.round((done / units.length) * 100) : 0,
  };
}

function nextUnitForRoute(routeId) {
  const units = routeUnits(routeId);
  return units.find((unit) => lessonState(unit.path).state !== 'done') || units[0] || null;
}

function renderRouteCard() {
  const container = $('#route-card');
  if (!container) return;
  const route = selectedRouteDefinition();
  const stats = routeProgress(route.id);
  const next = nextUnitForRoute(route.id);
  container.innerHTML = `<div>
      <p class="eyebrow">학습 경로</p>
      <strong>${escapeHtml(route.label)}</strong>
      <span>${escapeHtml(route.description)}</span>
    </div>
    <div>
      <strong>${stats.percent}%</strong>
      <span>${stats.done}/${stats.total} 완료 · 진행 중 ${stats.inProgress}</span>
      ${next ? `<button type="button" data-route-next="${escapeHtml(next.path)}">다음 단원 추천: ${escapeHtml(next.title)}</button>` : '<span>추천할 단원이 없습니다.</span>'}
    </div>`;
  container.querySelector('[data-route-next]')?.addEventListener('click', (event) => selectUnit(event.currentTarget.dataset.routeNext));
}

function learningStepsFor(unit) {
  const hasMlRunner = (unit.resources || []).some((resource) => resource.label === 'run_stage.py');
  const steps = hasMlRunner ? [
    { label: '이론 읽기', description: 'README / THEORY로 데이터셋, baseline, metric의 역할을 먼저 잡는다.' },
    { label: '실습 구성', description: 'dataset.py와 experiment.py에서 데이터 생성, 모델, 평가 지표가 어떻게 연결되는지 본다.' },
    { label: '실행 명령', description: 'run_stage.py를 기준으로 어떤 stage가 어떤 산출물을 만드는지 확인한다.' },
    { label: 'analysis 정리', description: 'analysis.py와 report.py로 결과를 해석하고 다음 실험 질문을 남긴다.' },
  ] : [
    { label: '이론 읽기', description: 'README / THEORY / PREREQS로 왜 배우는지와 선행 개념을 잡는다.' },
    { label: 'scratch 실행', description: 'scratch_lab.py로 작은 수치와 직접 계산을 확인한다.' },
    { label: 'framework 실행', description: 'framework_lab.py로 PyTorch나 프레임워크 관측을 비교한다.' },
    { label: 'analysis 정리', description: 'analysis.py와 analysis.md로 관측값을 한국어 해석으로 남긴다.' },
    { label: 'reflection 작성', description: 'reflection.md에 헷갈린 점, 실패 사례, 다음 질문을 적는다.' },
  ];
  return steps.filter((step) => {
    if (step.label.includes('scratch')) return unit.checkpoints.includes('scratch lab');
    if (step.label.includes('framework')) return unit.checkpoints.includes('framework lab');
    if (step.label.includes('실습 구성')) return unit.checkpoints.includes('실습 구성');
    if (step.label.includes('실행 명령')) return unit.checkpoints.includes('실행 명령');
    if (step.label.includes('analysis')) return unit.checkpoints.includes('analysis script') || unit.checkpoints.includes('analysis note');
    if (step.label.includes('reflection')) return unit.checkpoints.includes('reflection');
    return true;
  });
}

function selfCheckProgress(items, checked) {
  const total = items.length;
  const done = items.filter((item) => checked?.[item.id]).length;
  return {
    total,
    done,
    percent: total ? Math.round((done / total) * 100) : 0,
  };
}

function selfChecksFor(unit) {
  const checks = [
    {
      id: 'goal',
      label: '이 단원의 목표를 한 문장으로 설명할 수 있다',
      hint: unit.objective || 'README 첫 단락을 자기 말로 바꿔 보세요.',
    },
    {
      id: 'run-observe',
      label: '코드 실행 결과에서 봐야 할 숫자와 산출물을 짚을 수 있다',
      hint: (unit.required_outputs || []).slice(0, 2).join(', ') || 'metrics, figure, analysis.md 중 무엇이 남는지 확인하세요.',
    },
  ];
  (unit.analysis_questions || []).slice(0, 2).forEach((question, index) => {
    checks.push({
      id: `analysis-${index + 1}`,
      label: `분석 질문에 답할 수 있다: ${question}`,
      hint: '실행 관찰 카드와 analysis 문서를 보고 2~3문장으로 답해 보세요.',
    });
  });
  if (unit.key_terms?.length) {
    checks.push({
      id: 'terms',
      label: `${unit.key_terms.slice(0, 3).join(', ')}를 구분해서 설명할 수 있다`,
      hint: '헷갈리는 용어는 내 메모에 남기고 다음 단원으로 넘어가세요.',
    });
  }
  return checks.slice(0, 5);
}

function studyLinksFor(unit) {
  const links = [
    { href: '../docs/02_study_guide.md', label: '학습 안내', reason: '무기초 → LLM/RLHF/Multimodal/VLA 경로 확인' },
  ];
  if (unit.path.startsWith('01_ml/') || unit.path.startsWith('02_deep_learning/')) {
    links.push({ href: '../docs/04_feature_matrix_to_neural_training_bridge.md', label: 'ML→DL 연결 문서', reason: '이 ML stage를 끝내고 딥러닝으로 넘어갈 때 읽기' });
  }
  if (unit.path.startsWith('05_advanced_nlp_llm/')) {
    links.push({ href: '../docs/06_decoder_generation_bridge.md', label: 'Decoder 생성 연결', reason: 'autoregressive decoding, sampling, prompt serialization, KV-cache 연결' });
  }
  if (unit.path.includes('05_advanced_nlp_llm/06_rlhf')) {
    links.push({ href: '../docs/05_rl_primer_for_rlhf.md', label: 'RLHF용 RL 입문', reason: 'reward/policy/rollout/KL/PPO 선행 정리' });
  }
  if (unit.path.startsWith('08_multimodal_bridge/') || unit.path.startsWith('09_multimodal/') || unit.path.startsWith('10_vla/')) {
    links.push({ href: '../docs/07_multimodal_generation_bridge.md', label: '멀티모달 생성 연결', reason: 'retrieval에서 captioning/VQA cross-attention과 grounding failure로 넘어가기' });
  }
  if (unit.path.startsWith('10_vla/')) {
    links.push({ href: '../docs/08_rl_to_vla_bridge.md', label: 'RL→VLA 연결', reason: 'MDP, trajectory, behavior cloning, offline RL, action space design 구분' });
    links.push({ href: '../09_multimodal/README.md', label: '09 멀티모달 복습', reason: 'VQA에서 action grounding으로 넘어가기 전 복습' });
  }
  const trackReadme = `../${unit.path.split('/')[0]}/README.md`;
  links.push({ href: trackReadme, label: '트랙 안내', reason: '현재 트랙의 역할과 다음 연결 확인' });
  return links;
}

function scopeGateFor(unit) {
  if (!unit.path.startsWith('10_vla/')) return '';
  return `<aside class="scope-gate" aria-label="VLA 범위 확인">
    <strong>VLA 범위 확인</strong>
    <span>이 단원은 discrete action token과 safety gate를 다룹니다. 연속 제어, 로봇 동역학, 시뮬레이터 rollout, full offline RL은 별도 심화 주제입니다.</span>
  </aside>`;
}

function executionLabelFor(unit) {
  const labels = (unit.resources || []).map((resource) => resource.label);
  if (labels.includes('run_stage.py')) return 'ML stage 실습';
  if (labels.includes('scratch_lab.py') || labels.includes('framework_lab.py')) return '표준 실습';
  if (labels.some((label) => label.endsWith('.py'))) return '코드 읽기 실습';
  return '문서 학습';
}

function selectUnit(unitPath) {
  const unit = findUnit(unitPath);
  if (!unit) return;
  selectedUnitPath = unitPath;
  selectedTrackId = unitPath.split('/')[0];
  selectedResourceHref = '';
  const previous = lessonState(unitPath);
  currentUser().lessons[unitPath] = { ...previous, lastOpenedAt: new Date().toISOString() };
  updateActiveUI({ selectedUnit: unitPath, selectedTrack: selectedTrackId });
  persistProgress();
  render();
}

function trackStats(track) {
  const total = track.units.length;
  const done = track.units.filter((unit) => lessonState(unit.path).state === 'done').length;
  return { total, done, percent: total ? Math.round((done / total) * 100) : 0 };
}

function renderOverallProgress() {
  const route = selectedRouteDefinition();
  const stats = routeProgress(route.id);
  const profileName = currentUser().displayName || activeUserId;
  $('#overall-progress').textContent = `${stats.percent}% · ${profileName}`;
}

function completionPercent(checkpoints, checked, state) {
  if (state === 'done') return 100;
  if (!checkpoints.length) return 0;
  return Math.round((checkpoints.filter((item) => checked[item]).length / checkpoints.length) * 100);
}

function findUnit(unitPath) {
  for (const track of catalog.tracks) {
    const unit = track.units.find((candidate) => candidate.path === unitPath);
    if (unit) return unit;
  }
  return null;
}

function findLastOpenedUnit() {
  const lessons = currentUser().lessons;
  return Object.entries(lessons).sort((a, b) => String(b[1].lastOpenedAt || '').localeCompare(String(a[1].lastOpenedAt || '')))[0]?.[0] || catalog.tracks[0]?.units[0]?.path || '';
}

function addProfile() {
  const displayName = prompt('새 사용자 이름을 입력하세요. 이 이름도 로컬에만 저장됩니다.', '새 학습자');
  if (!displayName) return;
  const id = `local-${Date.now()}`;
  progressStore.users[id] = { displayName, lessons: {}, ui: Progress.defaultUI() };
  activeUserId = id;
  progressStore.activeUserId = id;
  restoreActiveView();
  persistProgress();
  render();
}

function exportProgress() {
  const blob = new Blob([JSON.stringify(progressStore, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'btb-local-progress.json';
  anchor.click();
  URL.revokeObjectURL(url);
}

function importProgress() {
  const raw = $('#import-json').value;
  try {
    const incoming = JSON.parse(raw);
    if (incoming.schemaVersion !== 1 || !incoming.users) throw new Error('schema mismatch');
    Progress.mergeImportedProgress(progressStore, incoming);
    activeUserId = progressStore.activeUserId;
    restoreActiveView();
    persistProgress();
    $('#import-dialog').close();
    render();
  } catch (error) {
    alert('가져오기 실패: BTB 진행 기록 JSON 형식을 확인하세요.');
  }
}

function resetProgress() {
  if (!confirm('현재 브라우저의 BTB 진행 기록을 삭제할까요? GitHub 데이터는 바뀌지 않습니다.')) return;
  progressStore = Progress.defaultProgress();
  activeUserId = progressStore.activeUserId;
  restoreActiveView();
  persistProgress();
  render();
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
