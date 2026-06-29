const Progress = window.BTBProgress;
const { STATES, STATE_LABELS } = Progress;

let catalog = { tracks: [] };
let selectedTrackId = '';
let selectedUnitPath = '';
let selectedResourceHref = '';
let progressStore = Progress.loadProgress();
let activeUserId = progressStore.activeUserId;
let contentRequestId = 0;


const SECTION_DISPLAY_LABELS = {
  README: '단원 안내',
  THEORY: '핵심 이론',
  PREREQS: '준비 확인',
  'scratch_lab.py': '기초 실습 코드',
  'framework_lab.py': '프레임워크 실습 코드',
  'analysis.py': '결과 해석 코드',
  'analysis.md': '해석 노트',
  'reflection.md': '회고 메모',
  'dataset.py': '데이터 준비 코드',
  'models.py': '모델 코드',
  'experiment.py': '실험 흐름 코드',
  'run_stage.py': '실험 실행 코드',
  'report.py': '리포트 코드',
};

const CHECKPOINT_DISPLAY_LABELS = {
  readme: '단원 안내',
  theory: '핵심 이론',
  prereqs: '준비 확인',
  'scratch lab': '기초 실습 실행',
  'framework lab': '프레임워크 실습 실행',
  'analysis script': '결과 해석 코드 실행',
  'analysis note': '해석 노트 확인',
  reflection: '회고 메모 작성',
  '실습 구성': '실습 구성 확인',
  '실행 명령': '실험 실행',
};

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
  $('#review-mistakes').addEventListener('click', openMistakeReview);
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
  $('#review-mistakes').textContent = `오답노트 ${allWrongNotes().length}`;
}


function displaySectionLabel(sectionOrLabel) {
  const raw = typeof sectionOrLabel === 'string' ? sectionOrLabel : sectionOrLabel?.label;
  return SECTION_DISPLAY_LABELS[raw] || raw || '자료';
}

function checkpointDisplayLabel(checkpoint) {
  const key = String(checkpoint || '').trim().toLowerCase();
  return CHECKPOINT_DISPLAY_LABELS[key] || checkpoint || '자료';
}

function unitStatusLabel(status) {
  if (status === 'runnable') return '실습 가능';
  if (status === 'planned') return '준비 중';
  if (status === 'partial') return '보강 중';
  return status || '상태 확인 전';
}

function humanUnitPath(unitPath) {
  return String(unitPath || '')
    .split('/')
    .filter(Boolean)
    .map((part) => part.replace(/^\d+_?/, '').replaceAll('_', ' '))
    .map((part) => part.replace(/\b\w/g, (letter) => letter.toUpperCase()))
    .join(' › ');
}

function friendlyDocumentLabel(label, href) {
  const raw = String(label || '').trim();
  const file = String(href || '').split('#')[0].split('/').pop();
  if (/^readme(\.md)?$/i.test(raw) || /^README\.md$/i.test(file)) return '단원 안내';
  if (/^theory(\.md)?$/i.test(raw) || /^THEORY\.md$/i.test(file)) return '핵심 이론';
  if (/^prereqs(\.md)?$/i.test(raw) || /^PREREQS\.md$/i.test(file)) return '준비 확인';
  return raw;
}


function documentSourceLabel(section) {
  const label = displaySectionLabel(section);
  if (label === '단원 안내') return '목표와 진행 안내';
  if (label === '핵심 이론') return '개념 설명';
  if (label === '준비 확인') return '선행 점검';
  if (section?.type === 'code') return '읽고 바로 실행';
  if (label === '해석 노트') return '결과 정리';
  if (label === '회고 메모') return '내 생각 정리';
  return '참고 자료';
}

function displayOutputLabel(item) {
  const raw = String(item || '').trim();
  const lower = raw.toLowerCase();
  if (!raw) return '학습 산출물';
  if (lower.includes('scratch') && (lower.includes('metric') || lower.includes('json'))) return '기초 실습 지표';
  if (lower.includes('framework') && (lower.includes('metric') || lower.includes('json'))) return '프레임워크 지표';
  if (lower.includes('stage') && (lower.includes('metric') || lower.includes('json'))) return '단계별 실험 지표';
  if (lower.includes('analysis') && (lower.includes('markdown') || lower.includes('.md') || lower.includes('report'))) return '해석 노트';
  if (lower.includes('summary') || raw.includes('실행 요약') || raw.includes('실습 요약')) return '실행 요약';
  if (lower.includes('prediction')) return '예측 샘플';
  if (lower.includes('figure') || lower.includes('svg')) return '그림/도표';
  if (lower.includes('config')) return '실험 설정';
  if (lower.includes('metric') || lower.includes('json')) return '지표 파일';
  if (lower.includes('report')) return '리포트';
  return raw;
}

function displayOutputList(items) {
  return (items || []).map(displayOutputLabel);
}

function sectionIsComplete(section, checked) {
  return Boolean(section?.checkpoint && checked?.[section.checkpoint]);
}

function renderSectionTab(section, selectedSection, checked) {
  const complete = sectionIsComplete(section, checked);
  return `<button type="button" role="tab" data-section-href="${escapeHtml(section.href)}" data-complete="${complete}" aria-selected="${hrefEquals(section.href, selectedSection.href)}" title="${complete ? '읽음 표시됨' : '아직 읽음 표시 전'}">${escapeHtml(displaySectionLabel(section))}${complete ? '<span class="tab-done-mark" aria-hidden="true">✓</span>' : ''}</button>`;
}

function renderTracks() {
  trackList.innerHTML = catalog.tracks.map((track) => {
    const stats = trackStats(track);
    return `<button class="track-card" type="button" aria-pressed="${track.id === selectedTrackId}" data-track="${escapeHtml(track.id)}">
      <div class="track-top"><span class="track-title">${escapeHtml(track.title)}</span><span class="track-meta">${stats.done}/${stats.total}</span></div>
      <div class="track-meta">${escapeHtml(track.id)}</div>
      <p>${renderInlineSummary(track.summary || '이 트랙에서 무엇을 익히는지 확인합니다.')}</p>
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
  const outputChips = displayOutputList(unit.required_outputs || []).slice(0, 3).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join('');
  return `<button class="unit-card" type="button" data-unit="${escapeHtml(unit.path)}" aria-current="${unit.path === selectedUnitPath}">
    <div class="unit-top"><span class="unit-title">${escapeHtml(unit.title)}</span><span class="chip ${progress.state}">${STATE_LABELS[progress.state]}</span></div>
    <div class="unit-meta">${escapeHtml(humanUnitPath(unit.path))} · ${escapeHtml(unitStatusLabel(unit.status))}</div>
    <p>${renderInlineSummary(unit.objective || '단원 안내에서 목표를 확인하세요.')}</p>
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
  const quizItems = quizForUnit(unit);
  const quizAnswers = progress.quizAnswers || {};
  const wrongNotes = progress.wrongNotes || {};
  const percent = completionPercent(checkpoints, checked, progress.state);
  const sections = lessonSectionsFor(unit);
  const selectedSection = sections.find((section) => hrefEquals(section.href, selectedResourceHref)) || sections[0];
  selectedResourceHref = selectedSection.href;

  detail.innerHTML = `<section class="lesson-hero">
      <div>
        <h2 id="detail-title">${escapeHtml(unit.title)}</h2>
        <p class="unit-meta">진행: ${escapeHtml(STATE_LABELS[progress.state])} · 방식: ${escapeHtml(executionLabelFor(unit))} · 위치: ${escapeHtml(humanUnitPath(unit.path))}</p>
        <p>${renderInlineSummary(unit.objective || '')}</p>
        ${scopeGateFor(unit)}
      </div>
      <div>
        <div class="next-action-card">${nextActionFor(unit, progress, selfCheckStats, quizItems, quizAnswers, checkpoints, checked)}</div>
        <div class="status-buttons" aria-label="진행 상태 변경">
          ${STATES.map((state) => `<button type="button" data-state="${state}" class="${state === progress.state ? 'active' : ''}">${STATE_LABELS[state]}</button>`).join('')}
        </div>
        <div class="progress-bar" aria-label="체크리스트 ${percent}% 완료"><span style="width:${percent}%"></span></div>
      </div>
    </section>
    <div class="lesson-workspace reader-shell">
      <aside class="lesson-guide" aria-label="학습 진행 가이드">
        ${renderLessonGuidePlan(unit)}
        ${prerequisiteReadinessFor(unit)}
        <h3>체크리스트</h3>
        <ul class="checklist">
          ${checkpoints.map((item) => `<li><label><input type="checkbox" data-checkpoint="${escapeHtml(item)}" ${checked[item] ? 'checked' : ''}/> ${escapeHtml(checkpointDisplayLabel(item))}</label></li>`).join('')}
        </ul>
        <h3>선행 확인</h3>
        <ul>${(unit.prereqs || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>이전 트랙과 study guide를 먼저 확인한다.</li>'}</ul>
        <h3>학습 방향</h3>
        <div class="resource-list">${studyLinksFor(unit).map((link) => `<button type="button" class="resource-button" data-resource-href="${escapeHtml(link.href)}" data-resource-label="${escapeHtml(link.label)}">${escapeHtml(link.label)}<span>${escapeHtml(link.reason)}</span></button>`).join('')}</div>
        <h3>핵심 용어</h3>
        <div class="chips">${(unit.key_terms || []).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join('') || '<span class="chip">단원 안내 참고</span>'}</div>
        <h3>남길 산출물</h3>
        <ul>${displayOutputList(unit.required_outputs || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>단원 안내와 결과 해석을 확인한다.</li>'}</ul>
        <h3>분석 질문</h3>
        <ul>${(unit.analysis_questions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('') || '<li>이 단원이 다음 트랙과 어떻게 연결되는지 설명한다.</li>'}</ul>
        <h3 class="self-check-heading">자가 점검 <span data-self-check-summary>${selfCheckStats.done}/${selfCheckStats.total} 완료</span></h3>
        <div class="self-check-meter" aria-label="자가 점검 ${selfCheckStats.percent}% 완료"><span style="width:${selfCheckStats.percent}%"></span></div>
        <ul class="self-checklist">
          ${selfCheckItems.map((item) => `<li><label><input type="checkbox" data-self-check="${escapeHtml(item.id)}" ${selfChecks[item.id] ? 'checked' : ''}/> ${escapeHtml(item.label)}</label><span>${escapeHtml(item.hint)}</span></li>`).join('')}
        </ul>
        ${renderQuizPanel(unit, quizItems, quizAnswers, wrongNotes)}
        ${renderWrongNotesPanel(unit, wrongNotes)}
        <h3>내 메모</h3>
        <textarea class="notes" id="unit-note" placeholder="헷갈린 개념, 다시 볼 코드, 다음 질문을 적어 두세요. 이 브라우저에만 저장됩니다.">${escapeHtml(progress.note || '')}</textarea>
      </aside>
      <section class="lesson-reader" aria-live="polite">
        <div class="reader-header">
          <div>
            <p class="eyebrow">학습 자료</p>
            <h3>오늘 볼 자료</h3>
          </div>
          <button id="mark-section-complete" type="button">읽음으로 표시</button>
        </div>
        <div class="document-tabs" role="tablist" aria-label="단원 자료">
          ${sections.map((section) => renderSectionTab(section, selectedSection, checked)).join('')}
        </div>
        <article id="lesson-content" class="lesson-content"><p class="empty">자료를 불러오는 중입니다.</p></article>
      </section>
    </div>`;

  bindDetailEvents(unit, checkpoints, checked, selfChecks, progress.state, selectedSection);
  bindQuizEvents(unit, quizItems, quizAnswers, wrongNotes);
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
  detail.querySelectorAll('[data-prereq-unit]').forEach((button) => {
    button.addEventListener('click', () => selectUnit(button.dataset.prereqUnit));
  });
  detail.querySelectorAll('[data-prereq-href]').forEach((button) => {
    button.addEventListener('click', () => {
      const section = {
        id: `prereq-${button.dataset.prereqLabel}`,
        label: button.dataset.prereqLabel || '선행 문서',
        href: button.dataset.prereqHref,
        type: 'markdown',
        checkpoint: '',
      };
      selectedResourceHref = section.href;
      loadLessonSection(unit, section);
    });
  });
  $('#unit-note').addEventListener('change', (event) => updateLesson(unit.path, { note: event.target.value }));
}

function bindQuizEvents(unit, quizItems, quizAnswers, wrongNotes) {
  detail.querySelectorAll('[data-quiz-submit]').forEach((button) => {
    button.addEventListener('click', () => {
      const question = quizItems.find((item) => item.id === button.dataset.quizSubmit);
      if (!question) return;
      const answer = readQuizAnswer(question);
      const correct = question.type === 'short' ? null : isQuizCorrect(question, answer);
      const now = new Date().toISOString();
      const nextAnswers = {
        ...quizAnswers,
        [question.id]: { answer, correct, reviewOnly: question.type === 'short', submittedAt: now },
      };
      const nextWrongNotes = { ...wrongNotes };
      if (question.type === 'short') {
        if (nextWrongNotes[question.id]) nextWrongNotes[question.id] = { ...nextWrongNotes[question.id], recovered: true, recoveredAt: now };
      } else if (correct) {
        if (nextWrongNotes[question.id]) nextWrongNotes[question.id] = { ...nextWrongNotes[question.id], recovered: true, recoveredAt: now };
      } else {
        nextWrongNotes[question.id] = {
          id: question.id,
          unitPath: unit.path,
          unitTitle: unit.title,
          question: question.prompt,
          learnerAnswer: formatQuizAnswer(question, answer),
          correctAnswer: correctAnswerFor(question),
          explanation: question.explanation,
          memo: nextWrongNotes[question.id]?.memo || '',
          recovered: false,
          updatedAt: now,
        };
      }
      updateLesson(unit.path, { quizAnswers: nextAnswers, wrongNotes: nextWrongNotes, state: lessonState(unit.path).state === 'not_started' ? 'in_progress' : lessonState(unit.path).state });
    });
  });
  detail.querySelectorAll('[data-wrong-note-memo]').forEach((textarea) => {
    textarea.addEventListener('change', () => {
      const id = textarea.dataset.wrongNoteMemo;
      const nextWrongNotes = {
        ...wrongNotes,
        [id]: { ...wrongNotes[id], memo: textarea.value, updatedAt: new Date().toISOString() },
      };
      updateLesson(unit.path, { wrongNotes: nextWrongNotes });
    });
  });
}

function readQuizAnswer(question) {
  if (question.type === 'multi') {
    return Array.from(detail.querySelectorAll(`[data-quiz-id="${question.id}"]:checked`)).map((input) => input.value).sort();
  }
  if (question.type === 'short') {
    return detail.querySelector(`[data-quiz-id="${question.id}"]`)?.value.trim() || '';
  }
  return detail.querySelector(`[data-quiz-id="${question.id}"]:checked`)?.value || '';
}

function isQuizCorrect(question, answer) {
  const expected = question.options.filter((option) => option.correct).map((option) => option.id).sort();
  const actual = Array.isArray(answer) ? [...answer].sort() : [answer].filter(Boolean);
  return expected.length === actual.length && expected.every((value, index) => value === actual[index]);
}

function formatQuizAnswer(question, answer) {
  if (Array.isArray(answer)) return answer.map((id) => optionLabel(question, id)).join(', ') || '(선택 없음)';
  return question.type === 'short' ? (answer || '(빈 답변)') : optionLabel(question, answer) || '(선택 없음)';
}

function correctAnswerFor(question) {
  if (question.type === 'short') return question.expected || '핵심 용어를 자기 말로 설명한 짧은 답변';
  return question.options.filter((option) => option.correct).map((option) => option.label).join(', ');
}

function optionLabel(question, id) {
  return question.options?.find((option) => option.id === id)?.label || id;
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

  const sectionLabel = displaySectionLabel(section);
  content.innerHTML = `<p class="empty">${escapeHtml(sectionLabel)} 자료를 사이트 안으로 불러오는 중입니다.</p>`;
  updateMarkSectionButton(unit, section, markButton);
  try {
    const text = await fetchLessonDocument(section.href);
    if (requestId !== contentRequestId) return;
    if (section.type === 'code') {
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(sectionLabel)}</span><span class="source-badge">${escapeHtml(documentSourceLabel(section))}</span></div>${renderCodeExplanation(section, text)}${renderCoreCodeSummary(section, text, unit)}<pre class="code-block"><code>${escapeHtml(annotateCodeWithInlineHints(section, text))}</code></pre>${renderRunPanel(section, unit, text)}`;
      bindRunButton(section, unit);
      bindCellProbeButton(section, unit);
    } else {
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(sectionLabel)}</span><span class="source-badge">${escapeHtml(documentSourceLabel(section))}</span></div>${renderMarkdown(text, section.href)}`;
      bindInlineDocLinks(unit, section.href);
    }
  } catch (error) {
    if (requestId !== contentRequestId) return;
    content.innerHTML = `<p class="empty">${escapeHtml(sectionLabel)}을 사이트 안에서 불러오지 못했습니다. 저장소 루트에서 <code>python -m http.server 8000</code>을 실행했는지 확인하세요.<br><code>${escapeHtml(cleanHref(section.href))}</code></p>`;
  }
}

async function fetchLessonDocument(href) {
  const response = await fetch(href, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`document load failed: ${response.status}`);
  return response.text();
}

function renderRunPanel(section, unit, source = '') {
  if (!isRunnableCodeSection(section, unit)) return '';
  const sourcePath = cleanHref(section.href);
  const runPath = runnablePathForSection(section, unit);
  const mappedToStage = runPath && runPath !== sourcePath;
  const plan = runPlanFor(section, unit);
  const symbols = extractPythonSymbols(source).map((symbol) => symbol.replace(/\(\)$/, ''));
  return `<section class="run-panel" aria-label="Python 코드 실행">
    <div>
      <p class="eyebrow">읽은 뒤 실행</p>
      <h4>이 코드를 내 환경에서 확인하기</h4>
      <p>위 코드를 먼저 훑은 다음 실행해 보세요. 종료 코드, 선택된 CPU/GPU, 출력과 산출물이 아래에 정리됩니다.</p>
      ${mappedToStage ? `<p class="run-target-note">이 탭은 실험의 일부입니다. 버튼은 같은 단원의 <code>${escapeHtml(runPath)}</code>를 실행해 데이터 준비·모델 학습·평가 결과를 함께 만듭니다.</p>` : ''}
    </div>
    <div class="run-actions">
      <button type="button" data-run-code data-run-path="${escapeHtml(runPath)}" data-run-source-path="${escapeHtml(sourcePath)}">${escapeHtml(runButtonLabel(section, unit))}</button>
      <span class="run-status" data-run-status>아직 실행 전입니다.</span>
    </div>
    <div class="run-primer" aria-label="실행 전 확인">
      <strong>실행 전에 볼 것</strong>
      <dl>
        <div><dt>예상 산출물</dt><dd>${escapeHtml(displayOutputList(plan.artifacts).join(', '))}</dd></div>
        <div><dt>봐야 할 숫자</dt><dd>${escapeHtml(plan.metrics.join(', '))}</dd></div>
        <div><dt>좋은 결과 기준</dt><dd>${escapeHtml(plan.goodOutcome)}</dd></div>
      </dl>
    </div>
    <div class="cell-probe" aria-label="선택 함수 미리보기">
      <strong>선택 함수 미리보기</strong>
      <p>전체 파일을 돌리기 전에 선택 함수의 입력·호출·산출물 단서를 안전하게 분석합니다. 임의 코드는 실행하지 않습니다.</p>
      <div class="cell-actions">
        <select data-cell-symbol aria-label="분석할 함수">
          ${symbols.length ? symbols.map((symbol) => `<option value="${escapeHtml(symbol)}">${escapeHtml(symbol)}()</option>`).join('') : '<option value="">파일 전체 구조</option>'}
        </select>
        <button type="button" data-run-cell>함수 구조 보기</button>
      </div>
      <div class="cell-output" data-cell-output hidden></div>
    </div>
    <div class="run-insights" data-run-insights hidden></div>
    <div class="artifact-viewer" data-artifact-viewer hidden></div>
    <pre class="run-output" data-run-output hidden></pre>
  </section>`;
}

function isRunnableCodeSection(section, unit = null) {
  return section?.type === 'code' && Boolean(runnablePathForSection(section, unit));
}

function runnablePathForSection(section, unit = null) {
  const path = cleanHref(section?.href || '');
  if (isDirectRunnableCodePath(path)) return path;
  if (isMlStageHelperSection(section, unit)) return stageRunnerForUnit(unit);
  return '';
}

function isDirectRunnableCodePath(path) {
  return /(?:scratch_lab|framework_lab|analysis|run_stage)\.py$/.test(cleanHref(path));
}

function isMlStageHelperSection(section, unit = null) {
  const path = cleanHref(section?.href || '');
  return section?.type === 'code'
    && path.startsWith('01_ml/')
    && /(?:dataset|models|experiment|report)\.py$/.test(path)
    && Boolean(stageRunnerForUnit(unit));
}

function stageRunnerForUnit(unit = null) {
  const runner = (unit?.resources || [])
    .map((resource) => repoRelativeResourcePath(resource.href || ''))
    .find((href) => href.endsWith('/run_stage.py'));
  return runner || '';
}

function repoRelativeResourcePath(href) {
  return String(href || '').trim().replace(/^\/+/, '').replace(/^(?:\.\.?\/)+/, '');
}

function effectiveRunPath(section, unit = null) {
  return runnablePathForSection(section, unit) || cleanHref(section?.href || '');
}

function runButtonLabel(section, unit = null) {
  const sourcePath = cleanHref(section?.href || '');
  const runPath = runnablePathForSection(section, unit);
  if (runPath && runPath !== sourcePath) return '전체 ML 실험 실행';
  return `${displaySectionLabel(section)} 실행`;
}

function bindRunButton(section, unit) {
  const button = $('#lesson-content [data-run-code]');
  if (!button) return;
  button.addEventListener('click', () => runPythonSection(section, button, unit));
}

function bindCellProbeButton(section, unit) {
  const button = $('#lesson-content [data-run-cell]');
  if (!button) return;
  button.addEventListener('click', () => runCodeCellProbe(section, button, unit));
}

async function runPythonSection(section, button, unit) {
  const panel = button.closest('.run-panel');
  const output = panel?.querySelector('[data-run-output]');
  const status = panel?.querySelector('[data-run-status]');
  const insights = panel?.querySelector('[data-run-insights]');
  const artifactViewer = panel?.querySelector('[data-artifact-viewer]');
  if (!output || !status) return;

  button.disabled = true;
  output.hidden = false;
  if (insights) {
    insights.hidden = true;
    insights.innerHTML = '';
  }
  if (artifactViewer) {
    artifactViewer.hidden = true;
    artifactViewer.innerHTML = '';
  }
  output.textContent = '실행 중입니다...';
  status.textContent = '실행 중';
  try {
    const response = await fetch('/api/run-python', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: button.dataset.runPath || runnablePathForSection(section, unit) || cleanHref(section.href) }),
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
    if (artifactViewer) {
      artifactViewer.innerHTML = renderArtifactViewer(payload, section, unit);
      artifactViewer.hidden = false;
    }
    status.textContent = payload.returncode === 0 ? '실행 완료' : `종료 코드 ${payload.returncode}`;
  } catch (error) {
    output.textContent = staticServerHelp(error.message);
    status.textContent = '실행 서버 필요';
  } finally {
    button.disabled = false;
  }
}

async function runCodeCellProbe(section, button, unit) {
  const panel = button.closest('.run-panel');
  const output = panel?.querySelector('[data-cell-output]');
  const select = panel?.querySelector('[data-cell-symbol]');
  if (!output) return;
  button.disabled = true;
  output.hidden = false;
  output.innerHTML = '<p class="empty">선택 셀을 분석하는 중입니다...</p>';
  try {
    const response = await fetch('/api/partial-experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: cleanHref(section.href), symbol: select?.value || '' }),
    });
    const payload = response.headers.get('content-type')?.includes('application/json')
      ? await response.json()
      : { error: await response.text(), status: response.status };
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    output.innerHTML = renderCellProbe(payload, unit);
  } catch (error) {
    output.textContent = staticServerHelp(error.message);
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
  const path = effectiveRunPath(section, unit);
  const declared = (unit?.required_outputs || []).filter((item) => !/^runnable README|theory note|prerequisite checklist$/i.test(item));
  if (path.endsWith('run_stage.py')) return declared.length ? declared : ['artifacts/<timestamp>/metrics.json', 'figures/', 'predictions/', 'summary.md'];
  if (path.endsWith('analysis.py')) return ['해석 노트', '관찰 지표', ...declared.filter((item) => /analysis|report|observed/i.test(item))].slice(0, 4);
  if (path.endsWith('framework_lab.py')) return declared.filter((item) => /framework|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['프레임워크 실습 요약']).slice(0, 4);
  if (path.endsWith('scratch_lab.py')) return declared.filter((item) => /scratch|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['기초 실습 요약']).slice(0, 4);
  return declared.length ? declared.slice(0, 4) : ['지표 파일', 'figure 또는 markdown report'];
}

function importantNumbersForRun(section, unit) {
  const path = effectiveRunPath(section, unit);
  const terms = unit?.key_terms || [];
  if (path.endsWith('run_stage.py')) return ['주요 평가 지표', '기준 모델 대비 좋은 모델', '학습/평가 데이터 수'];
  if (path.endsWith('analysis.py')) return ['빠진 결과물 수', '실패 사례 수', '해석 노트가 강조한 핵심 지표'];
  if (path.endsWith('framework_lab.py') && unit?.path === '00_foundations/04_regularization_and_normalization') {
    return ['data_loss_before_step은 같을 수 있음', 'regularized_objective_before_step', 'post_step_data_loss와 weight_norm_after_step'];
  }
  if (path.endsWith('framework_lab.py')) return ['loss 또는 accuracy 추세', '기초 실습과 같은 모양/지표인지', '실행 장치와 재실행 기준값'];
  if (path.endsWith('scratch_lab.py')) return ['입력/출력 모양', '핵심 계산 결과', terms[0] ? `${terms[0]} 관측값` : '작은 예제 지표'];
  return ['종료 코드', '지표', '결과물 위치'];
}

function goodOutcomeForRun(section, unit) {
  const path = effectiveRunPath(section, unit);
  const deterministic = unit?.deterministic ? ' 같은 설정으로 재실행해도 핵심 숫자가 유지되어야 합니다.' : '';
  if (path.endsWith('run_stage.py')) return `종료 코드 0, 지표·그림·예측 샘플이 생기고 단원 안내의 기준 모델 질문에 답할 수 있으면 좋습니다.${deterministic}`;
  if (path.endsWith('analysis.py')) return `기초 실습 코드와 프레임워크 실습 코드를 먼저 실행한 뒤, 이전 실행 결과물을 빠짐없이 읽고 해석 노트에 실패 사례와 다음 실험 질문이 남으면 좋습니다.${deterministic}`;
  if (path.endsWith('framework_lab.py') && unit?.path === '00_foundations/04_regularization_and_normalization') {
    return `step 전 data loss가 같아도 regularized objective, effective gradient, post-step loss, weight norm에서 decay 효과가 분리되어 보이면 좋습니다.${deterministic}`;
  }
  if (path.endsWith('framework_lab.py')) return `프레임워크 결과가 기초 실습 기준선과 설명 가능한 차이만 보이고, 실행 환경과 재실행 기준값이 출력에 남으면 좋습니다.${deterministic}`;
  if (path.endsWith('scratch_lab.py')) return `작은 입력에서 모양과 계산값을 직접 설명할 수 있고, 지표 파일/그림이 해석 기준선으로 남으면 좋습니다.${deterministic}`;
  return `종료 코드 0과 다시 확인할 수 있는 결과물 위치가 남으면 좋습니다.${deterministic}`;
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
    <ul>${plan.artifacts.map((item) => `<li>${escapeHtml(displayOutputLabel(item))}</li>`).join('')}</ul>
    <h5>봐야 할 숫자</h5>
    <ul>${highlights.length ? highlights.map((item) => `<li><code>${escapeHtml(item.path)}</code>: ${escapeHtml(item.value)}</li>`).join('') : plan.metrics.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    <h5>좋은 결과 기준</h5>
    <p>${escapeHtml(plan.goodOutcome)}</p>
    <h5>다음 질문</h5>
    <ul>${nextQuestions.map((question) => `<li>${escapeHtml(question)}</li>`).join('')}</ul>
  </section>`;
}

function renderArtifactViewer(payload, section, unit) {
  const artifacts = Array.isArray(payload.artifacts) ? payload.artifacts : [];
  const expected = expectedArtifactsForRun(section, unit);
  const missing = expected.filter((item) => !artifacts.some((artifact) => artifact.path?.toLowerCase().includes(keywordForArtifact(item))));
  return `<section aria-label="산출물 뷰어">
    <p class="eyebrow">산출물 뷰어</p>
    <h4>실행 산출물 바로 보기</h4>
    <p>이번 실행에서 새로 만들어지거나 갱신된 지표, 그림, 표, 분석 노트를 먼저 확인하세요.</p>
    ${missing.length ? `<div class="artifact-missing"><strong>확인 필요</strong><span>예상 결과물 중 아직 보이지 않는 항목: ${escapeHtml(displayOutputList(missing).slice(0, 3).join(', '))}. 전체 실행 순서나 해석 단계에 필요한 결과물을 확인하세요.</span></div>` : ''}
    <div class="artifact-grid">
      ${artifacts.length ? artifacts.map((artifact) => renderArtifactCard(artifact)).join('') : '<p class="empty">이번 실행에서 새로 갱신된 결과물이 없습니다. 파일이 읽기 전용이거나, 선행 실험 결과물이 필요한 해석 단계일 수 있습니다.</p>'}
    </div>
  </section>`;
}

function keywordForArtifact(label) {
  const text = String(label || '').toLowerCase();
  if (text.includes('scratch')) return 'scratch';
  if (text.includes('framework')) return 'framework';
  if (text.includes('analysis') || text.includes('report')) return 'analysis';
  if (text.includes('figure') || text.includes('svg')) return '.svg';
  if (text.includes('metric') || text.includes('json')) return '.json';
  return text.split(/[\\s/_.-]+/).find((part) => part.length > 3) || text;
}

function artifactLabel(artifact) {
  const path = String(artifact.path || '').toLowerCase();
  if (path.endsWith('.svg')) return '결과 그림';
  if (path.endsWith('.csv')) return '예측/샘플 표';
  if (path.endsWith('.md')) return '분석 노트';
  if (path.endsWith('.json') && path.includes('metric')) return '지표 요약';
  if (path.endsWith('.json')) return '구조화 결과';
  return '실행 산출물';
}

function renderArtifactCard(artifact) {
  const preview = artifact.preview || {};
  return `<article class="artifact-card">
    <div class="artifact-head">
      <strong>${escapeHtml(artifactLabel(artifact))}</strong>
      <code>${escapeHtml(artifact.path || '')}</code>
    </div>
    <p>${escapeHtml(artifact.type || 'file')} · ${escapeHtml(formatBytes(artifact.size_bytes || 0))}</p>
    ${renderArtifactPreview(preview)}
  </article>`;
}

function formatBytes(size) {
  const bytes = Number(size) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderArtifactPreview(preview) {
  if (preview.kind === 'json') {
    return `<pre class="artifact-json">${escapeHtml(JSON.stringify(preview.json, null, 2).slice(0, 5000))}</pre>`;
  }
  if (preview.kind === 'svg' && preview.data_uri) {
    return `<img class="artifact-image" alt="실행 산출물 SVG 미리보기" src="${escapeHtml(preview.data_uri)}" />`;
  }
  if (preview.kind === 'text') {
    const text = String(preview.text || '');
    if (text.includes(',') && text.includes('\n')) return renderCsvPreview(text);
    return `<pre class="artifact-text">${escapeHtml(text.slice(0, 3000))}</pre>`;
  }
  return `<p class="empty">${escapeHtml(preview.message || '이 파일은 아직 브라우저 미리보기를 지원하지 않습니다.')}</p>`;
}

function renderCsvPreview(text) {
  const rows = text.trim().split('\n').slice(0, 7).map((line) => line.split(',').slice(0, 6));
  if (!rows.length) return '<p class="empty">표 미리보기를 만들 수 없습니다.</p>';
  const [head, ...body] = rows;
  return `<div class="table-wrap"><table class="artifact-table">
    <thead><tr>${head.map((cell) => `<th>${escapeHtml(cell)}</th>`).join('')}</tr></thead>
    <tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
  </table></div>`;
}

function renderCellProbe(payload, unit) {
  const cell = payload.cell || {};
  if (cell.mode === 'module_probe') {
    return `<section>
      <p class="eyebrow">선택 함수 미리보기</p>
      <h5>모듈 구조 분석</h5>
      <p>${escapeHtml(cell.learning_note || '')}</p>
      <p><strong>산출물 단서:</strong> ${escapeHtml((cell.artifact_names || []).join(', ') || '상단 경로 변수를 확인하세요.')}</p>
    </section>`;
  }
  return `<section>
    <p class="eyebrow">선택 함수 미리보기</p>
    <h5>${escapeHtml(cell.signature || '선택 함수')}</h5>
    <p>${escapeHtml(cell.learning_note || '')}</p>
    <dl class="cell-facts">
      <div><dt>줄 범위</dt><dd>${escapeHtml((cell.line_range || []).join('–'))}</dd></div>
      <div><dt>호출하는 이름</dt><dd>${escapeHtml((cell.called_names || []).join(', ') || '직접 계산')}</dd></div>
      <div><dt>중간 변수</dt><dd>${escapeHtml((cell.local_variables || []).join(', ') || '상단 설정값 중심')}</dd></div>
      <div><dt>산출물 단서</dt><dd>${escapeHtml((cell.artifact_names || []).join(', ') || (unit.required_outputs || []).slice(0, 2).join(', ') || '실행 후 산출물 확인')}</dd></div>
    </dl>
    <h5>작게 읽어볼 코드</h5>
    <pre class="artifact-text">${escapeHtml(cell.source_excerpt || '')}</pre>
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
  const path = effectiveRunPath(section, unit);
  if (path.endsWith('analysis.py') && Number(payload.returncode) !== 0) {
    const text = `${payload.stdout || ''}\n${payload.stderr || ''}`;
    if (text.includes('필수 metrics 파일이 없습니다')) {
      return '기초/프레임워크 metrics가 빠졌습니다. 서버 재시작이 자동 삭제하지는 않지만, scratch_lab.py와 framework_lab.py를 먼저 다시 실행하세요.';
    }
  }
  if (path.endsWith('analysis.py')) return '해석 노트나 관찰 지표가 갱신됐는지 확인하세요.';
  if (path.endsWith('run_stage.py')) return '결과 폴더에서 지표, 그림, 예측 샘플, 요약을 확인하세요.';
  if (path.endsWith('framework_lab.py')) return '프레임워크 지표와 그림을 기초 실습 결과와 나란히 비교하세요.';
  if (path.endsWith('scratch_lab.py')) return '기초 실습 지표 파일과 작은 표/그림이 해석의 기준선입니다.';
  const expected = expectedArtifactsForRun(section, unit)[0];
  if (payload.path) return `${payload.path} 실행 결과와 ${expected}를 확인하세요.`;
  return `실행 결과와 ${expected}를 확인하세요.`;
}

function runFollowupQuestions(section, payload, highlights, unit) {
  const path = effectiveRunPath(section, unit);
  const questions = [];
  if (Number(payload.returncode) !== 0) {
    if (path.endsWith('analysis.py')) {
      questions.push('분석 코드는 앞선 metrics를 읽는 단계입니다. scratch_lab.py와 framework_lab.py가 먼저 성공했는지 확인하세요.');
    } else {
      questions.push('오류 출력에서 missing file, dependency, timeout 중 무엇이 원인인지 분류하세요.');
    }
    questions.push('CPU/GPU/conda 환경을 바꿔 재실행해야 하는지 확인하세요.');
    return questions;
  }
  if (highlights.length) {
    questions.push('가장 중요한 숫자 하나를 단원 안내의 성공 기준이나 분석 질문과 연결해 설명해 보세요.');
  } else {
    questions.push('출력 원문에서 입력 모양, loss, accuracy, 저장 위치 중 무엇을 확인해야 하는지 표시해 보세요.');
  }
  if (unit?.analysis_questions?.[0]) questions.push(`분석 질문과 연결: ${unit.analysis_questions[0]}`);
  if (path.endsWith('scratch_lab.py')) questions.push('기초 실습 결과와 프레임워크 결과가 같아야 하는 부분과 달라도 되는 부분을 구분하세요.');
  else if (path.endsWith('framework_lab.py')) questions.push('프레임워크가 자동으로 처리한 부분이 기초 실습 코드의 어느 줄과 대응되는지 찾아보세요.');
  else if (path.endsWith('analysis.py')) questions.push('해석 노트가 말하는 실패 사례나 다음 실험 질문을 내 메모에 한 줄로 남기세요.');
  else if (path.endsWith('run_stage.py')) questions.push('데이터 준비와 실험 흐름 중 어떤 단계가 이 숫자에 가장 크게 영향을 줬는지 추적하세요.');
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

function renderCoreCodeSummary(section, source, unit = null) {
  const guide = coreCodeGuideFor(section, source, unit);
  if (!guide) return '';
  return `<section class="core-code-summary" aria-label="핵심 코드 먼저 보기">
    <div>
      <p class="eyebrow">핵심 코드 먼저 보기</p>
      <h4>${escapeHtml(guide.title)}</h4>
      <p>${escapeHtml(guide.summary)}</p>
    </div>
    <ol>
      ${guide.steps.map((step) => `<li>
        <strong>${escapeHtml(coreStepLabelText(step.label))}</strong>
        <span>${escapeHtml(step.note)}</span>
        ${step.code ? `<pre class="mini-code"><code>${escapeHtml(step.code)}</code></pre>` : ''}
      </li>`).join('')}
    </ol>
  </section>`;
}

function coreStepLabelText(label) {
  return String(label || '').replace(/^\s*\d+[.)]\s+/, '');
}

const CORE_CODE_GUIDE_OVERRIDES = {
  '00_foundations/03_gradients_and_backpropagation/scratch_lab.py': gradientBackpropCoreGuide,
  '00_foundations/04_regularization_and_normalization/scratch_lab.py': regularizationScratchCoreGuide,
  '00_foundations/04_regularization_and_normalization/framework_lab.py': regularizationFrameworkCoreGuide,
  '00_foundations/04_regularization_and_normalization/analysis.py': regularizationAnalysisCoreGuide,
};

function coreCodeGuideFor(section, source, unit = null) {
  const path = cleanHref(section.href);
  const lines = source.split('\n');
  if (CORE_CODE_GUIDE_OVERRIDES[path]) return CORE_CODE_GUIDE_OVERRIDES[path](source);
  if (path.endsWith('run_stage.py')) return runStageCoreGuide(source);
  if (path.startsWith('01_ml/') && path.endsWith('dataset.py')) return mlDatasetCoreGuide(source);
  if (path.startsWith('01_ml/') && path.endsWith('experiment.py')) return mlExperimentCoreGuide(source);
  if (!path.endsWith('.py')) return null;
  const symbols = extractPythonSymbolNames(source);
  const steps = automaticCoreCodeSteps(section, source, symbols, unit).slice(0, 4);
  if (!isLongOrDensePythonSource(lines, symbols) && steps.length < 2) return null;
  if (!steps.length) return null;
  return {
    title: coreCodeGuideTitleFor(path, lines.length),
    summary: coreCodeGuideSummaryFor(path, unit),
    steps,
  };
}

function gradientBackpropCoreGuide(source) {
  return {
    title: 'Gradient 실습은 이 네 덩어리만 먼저 읽으면 됩니다',
    summary: '전체 파일에는 그림 저장과 JSON 저장 코드도 섞여 있습니다. 처음에는 아래 계산 흐름만 보고, 나머지는 결과를 보기 좋게 남기는 주변 코드로 미뤄도 됩니다.',
    steps: [
      {
        label: '1. 예측값과 loss 만들기',
        note: '선형 모델의 출력과 정답 차이를 하나의 loss 숫자로 압축합니다.',
        code: compactCodeLines(source, [
          'prediction = (weight * x_value) + bias',
          'error = prediction - target',
          'loss = 0.5 * (error**2)',
        ]),
      },
      {
        label: '2. chain rule로 손미분 gradient 계산하기',
        note: 'loss가 prediction을 얼마나 밀어야 하는지 구한 뒤, weight와 bias 쪽으로 나눠 보냅니다.',
        code: compactCodeLines(source, [
          'dloss_dprediction = prediction - target',
          'grad_w = dloss_dprediction * x_value',
          'grad_b = dloss_dprediction',
        ]),
      },
      {
        label: '3. finite difference로 미분값 검산하기',
        note: '아주 작은 epsilon만큼 양쪽으로 움직여 loss 기울기를 근사하고, 손미분 결과와 비교합니다.',
        code: compactCodeLines(source, [
          'loss_plus = forward_loss(weight + epsilon, bias)',
          'loss_minus = forward_loss(weight - epsilon, bias)',
          'return (loss_plus - loss_minus) / (2.0 * epsilon)',
        ]),
      },
      {
        label: '4. gradient 방향으로 파라미터 업데이트하기',
        note: 'loss를 줄이는 방향으로 weight와 bias를 한 걸음 이동시킨 뒤, updated_loss가 줄었는지 확인합니다.',
        code: compactCodeLines(source, [
          'updated_weight = WEIGHT - (LEARNING_RATE * grad_w)',
          'updated_bias = BIAS - (LEARNING_RATE * grad_b)',
          'updated_prediction, updated_loss = forward_loss(updated_weight, updated_bias)',
        ]),
      },
    ],
  };
}

function isLongOrDensePythonSource(lines, symbols) {
  const hasFlowAnchor = symbols.some((name) => /run|main|train|evaluate|forward|fit|step|metric|loss|score/i.test(name));
  return lines.length >= 160 || (lines.length >= 110 && symbols.length >= 3 && hasFlowAnchor);
}

function regularizationScratchCoreGuide(source) {
  return {
    title: 'Normalization/Regularization 실습은 이 네 부분이 핵심입니다',
    summary: '그림 저장 코드는 뒤로 미루고, 입력 스케일을 바꾸는 정규화와 weight decay가 gradient·loss·weight norm에 들어가는 지점만 먼저 보세요.',
    steps: [
      {
        label: '1. z-score normalization으로 입력 스케일 맞추기',
        note: 'raw feature의 큰 숫자를 평균 0, 표준편차 1 근처로 바꿔 같은 learning rate가 과하게 튀지 않게 합니다.',
        code: compactCodeLines(source, [
          'centered = values - values.mean()',
          'return centered / values.std()',
          'normalized_features = zscore(RAW_FEATURES)',
        ]),
      },
      {
        label: '2. data loss와 L2 regularization loss 분리하기',
        note: 'loss 자체와 weight를 크게 만들지 않으려는 penalty를 따로 계산한 뒤 더합니다.',
        code: compactCodeLines(source, [
          'data_loss = 0.5 * float(np.mean(errors**2))',
          'reg_loss = 0.5 * weight_decay * (weight**2)',
          'total_loss = data_loss + reg_loss',
        ]),
      },
      {
        label: '3. gradient에 weight decay 항을 더하기',
        note: 'L2 regularization은 loss에만 숫자를 더하는 것이 아니라 weight gradient에도 weight_decay * weight를 추가합니다.',
        code: compactCodeLines(source, [
          'grad_w = float(np.mean(errors * features) + (weight_decay * weight))',
          'weight -= learning_rate * grad_w',
          'bias -= learning_rate * grad_b',
        ]),
      },
      {
        label: '4. raw / normalized / normalized+L2를 같은 조건에서 비교하기',
        note: '단원 결론은 세 실행의 log10(loss), gradient scale, final weight norm을 나란히 비교할 때 보입니다.',
        code: compactCodeLines(source, [
          'raw_run = run_training(RAW_FEATURES, TARGETS, learning_rate=LEARNING_RATE)',
          'normalized_run = run_training(normalized_features, TARGETS, learning_rate=LEARNING_RATE)',
          'weight_decay=WEIGHT_DECAY,',
        ]),
      },
    ],
  };
}

function regularizationFrameworkCoreGuide(source) {
  return {
    title: '프레임워크 실습은 LayerNorm·Dropout·Weight Decay를 따로 보세요',
    summary: 'PyTorch가 자동으로 처리하는 부분이 많아서, normalizing layer, train/eval mode, optimizer의 weight_decay 옵션을 분리해서 읽는 것이 핵심입니다.',
    steps: [
      {
        label: '1. LayerNorm이 행마다 평균/분산을 맞추는지 확인하기',
        note: 'LayerNorm은 batch 전체가 아니라 각 row의 마지막 차원 기준으로 normalize합니다.',
        code: compactCodeLines(source, [
          'layer_norm = torch.nn.LayerNorm(4, elementwise_affine=False, eps=0.0)',
          'normalized = layer_norm(inputs)',
          "'layernorm_row_means': _rounded_list(normalized.mean(dim=-1)),",
        ]),
      },
      {
        label: '2. Dropout은 train/eval mode에서 다르게 동작합니다',
        note: '학습 모드에서는 일부 값을 0으로 만들고, 평가 모드에서는 입력을 그대로 통과시키는 차이를 봅니다.',
        code: compactCodeLines(source, [
          'dropout.train()',
          'dropout_train = dropout(inputs)',
          'dropout.eval()',
          'dropout_eval = dropout(inputs)',
        ]),
      },
      {
        label: '3. optimizer의 weight_decay가 업데이트에 들어가는 지점',
        note: '프레임워크에서는 L2 항을 손으로 gradient에 더하지 않고 optimizer 옵션으로 전달합니다.',
        code: compactCodeLines(source, [
          'optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=weight_decay)',
          'loss.backward()',
          'optimizer.step()',
        ]),
      },
      {
        label: '4. 같은 data loss 뒤에 달라지는 objective와 step 결과 읽기',
        note: 'PyTorch의 data loss는 decay 유무가 같게 보일 수 있습니다. regularized objective, post-step loss, weight norm을 함께 봐야 합니다.',
        code: compactCodeLines(source, [
          'no_weight_decay = run_weight_decay_step(weight_decay=0.0)',
          'with_weight_decay = run_weight_decay_step(weight_decay=0.2)',
          "'weight_decay_regularized_objective_before_step': with_weight_decay['regularized_objective_before_step'],",
          "'weight_decay_post_step_data_loss': with_weight_decay['post_step_data_loss'],",
        ]),
      },
    ],
  };
}

function regularizationAnalysisCoreGuide(source) {
  return {
    title: 'Regularization 해석 코드는 지표를 결론으로 바꾸는 흐름입니다',
    summary: 'scratch/framework 실습 metrics를 다시 읽고, normalization 효과와 weight decay가 “loss는 같아 보여도 업데이트는 달라지는” 이유를 한국어 관측 리포트로 정리합니다.',
    steps: [
      {
        label: '필수 metrics가 있는지 먼저 확인하기',
        note: 'analysis.py는 단독 계산 파일이 아니라 앞선 두 실습의 결과를 해석합니다. 그래서 scratch/framework metrics가 없으면 바로 멈추고 무엇을 먼저 실행할지 알려줍니다.',
        code: compactCodeLines(source, [
          'missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]',
          'if not missing:',
          '    return',
          "missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)",
          'raise SystemExit(',
          "    '필수 metrics 파일이 없습니다: '",
          "    f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'",
          ')',
        ]),
      },
      {
        label: 'scratch와 framework 결과를 해석 입력으로 읽기',
        note: '정규화가 gradient scale을 바꾼 증거는 scratch metrics에서, LayerNorm·Dropout·Weight Decay의 프레임워크 관측은 framework metrics에서 가져옵니다.',
        code: compactCodeLines(source, [
          'scratch = _load_json(SCRATCH)',
          'framework = _load_json(FRAMEWORK)',
          "raw_initial_grad = float(scratch['raw_initial_grad_norm'])",
          "normalized_initial_grad = float(scratch['normalized_initial_grad_norm'])",
        ]),
      },
      {
        label: 'weight decay가 같아 보이는 이유를 분리해서 해석하기',
        note: 'step 전 data loss는 decay 유무와 무관하게 같을 수 있습니다. 대신 regularized objective, decay term, post-step data loss 차이를 함께 읽어야 결론이 보입니다.',
        code: compactCodeLines(source, [
          "data_loss_before = float(framework['weight_decay_data_loss_before_step'])",
          "no_decay_data_loss_before = float(framework['no_weight_decay_data_loss_before_step'])",
          "decay_objective = float(framework['weight_decay_regularized_objective_before_step'])",
          "post_step_delta = float(framework['post_step_data_loss_delta'])",
        ]),
      },
      {
        label: 'stable analysis와 이번 실행 리포트를 나눠 저장하기',
        note: 'analysis.md는 항상 같은 해석 프레임을 유지하고, 매번 달라질 수 있는 숫자 관측은 artifacts/analysis-manual/latest_report.md에 따로 남깁니다.',
        code: compactCodeLines(source, [
          'OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)',
          "OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')",
          "ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')",
          'print(observed_report)',
        ]),
      },
    ],
  };
}



function mlDatasetCoreGuide(source) {
  const symbols = extractPythonSymbolNames(source);
  const loadName = symbols.find((name) => /load|read/.test(name.toLowerCase())) || symbols[0];
  const preprocessName = symbols.find((name) => /preprocess|transform|feature/.test(name.toLowerCase()));
  const splitName = symbols.find((name) => /split|make/.test(name.toLowerCase()) && name !== loadName);
  const steps = [
    loadName && {
      label: `원본 표 데이터 읽기: ${loadName}()`,
      note: '실험의 출발점입니다. 어떤 원본 표를 읽고 결측/타입 같은 기본 정리를 하는지 먼저 확인합니다.',
      code: extractFunctionExcerpt(source, loadName),
    },
    preprocessName && {
      label: `feature 전처리 계약 만들기: ${preprocessName}()`,
      note: '숫자/범주형 feature를 어떤 transformer나 pipeline으로 바꾸는지 확인합니다.',
      code: extractFunctionExcerpt(source, preprocessName),
    },
    splitName && {
      label: `train/valid/test split 만들기: ${splitName}()`,
      note: '모든 모델 비교가 같은 데이터 분할과 target 정의를 쓰도록 고정하는 부분입니다.',
      code: extractFunctionExcerpt(source, splitName),
    },
  ].filter(Boolean);
  return {
    title: '데이터 준비 코드는 원본 표→전처리→split 계약만 먼저 보세요',
    summary: '모델보다 먼저 데이터 계약이 고정되어야 합니다. 어떤 표를 읽고, feature를 어떻게 바꾸며, train/valid/test가 어디서 나뉘는지 확인하세요.',
    steps,
  };
}

function mlExperimentCoreGuide(source) {
  return {
    title: '실험 흐름 코드는 split→모델 비교→best 선택→산출물 저장 순서로 보세요',
    summary: 'ML 단원의 experiment.py는 단일 모델 코드가 아니라 baseline, 학습 모델, GPU 모델, 지표/그림/CSV 저장을 한 stage 계약으로 묶는 파일입니다.',
    steps: [
      {
        label: 'stage context와 데이터 split 준비',
        note: '실험 이름, primary metric, device를 고정하고 같은 split을 모든 모델 비교에 사용합니다.',
        code: compactCodeLines(source, [
          "ctx = build_stage_context(",
          'split = make_split()',
        ]),
      },
      {
        label: 'baseline과 후보 모델을 같은 방식으로 학습·예측하기',
        note: '각 모델의 prediction, score, fit time, memory를 같은 ModelResult 구조로 모아 비교 가능하게 만듭니다.',
        code: compactCodeLines(source, [
          'for name, model in sklearn_models.items():',
          'model, y_pred, y_score, fit_time, predict_time, peak_rss = timed_fit_predict(model, split.X_train, split.y_train, split.X_test)',
          'results[name] = ModelResult(',
        ]),
      },
      {
        label: 'primary metric으로 best model 고르기',
        note: 'accuracy 하나가 아니라 단원에서 정한 primary metric 기준으로 대표 모델을 선택합니다.',
        code: compactCodeLines(source, [
          "best_name = max(results, key=lambda model_name: results[model_name].metrics[ctx.primary_metric])",
          'best = results[best_name]',
        ]),
      },
      {
        label: 'metrics, prediction CSV, figure를 저장하기',
        note: '웹사이트와 analysis.py가 다시 읽을 지표와 오류 사례, 시각화를 파일로 남깁니다.',
        code: compactCodeLines(source, [
          "json_dump(ctx.run_paths.run_dir / 'metrics.json', {",
          "to_csv(ctx.run_paths.predictions_dir /",
          'bar_chart(',
        ]),
      },
    ],
  };
}

function runStageCoreGuide(source) {
  return {
    title: '실험 실행 코드는 환경 선택→stage 호출→요약 출력만 먼저 보세요',
    summary: '이 파일은 모델 내용을 다시 구현하지 않습니다. GPU 번호와 실행 환경을 정하고, 같은 폴더의 experiment.run_stage(device)를 호출해 결과 JSON을 터미널에 요약합니다.',
    steps: [
      {
        label: '실행 옵션에서 GPU 번호 받기',
        note: '사용자가 넘긴 --gpu 값을 CUDA_VISIBLE_DEVICES 기본값으로 사용해 어느 장치를 볼지 정합니다.',
        code: compactCodeLines(source, [
          "parser.add_argument('--gpu', type=int, default=0)",
          'return parser.parse_args()',
        ]),
      },
      {
        label: 'seed와 device를 실행 직전에 확정하기',
        note: '재현성을 위해 seed를 고정하고, CUDA 사용 가능 여부에 따라 cuda/cpu 실행 경로를 선택합니다.',
        code: compactCodeLines(source, [
          "os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))",
          'set_seed()',
          "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        ]),
      },
      {
        label: 'stage 실험을 호출하고 JSON으로 출력하기',
        note: '실제 데이터 split, 모델 학습, metric/figure 저장은 experiment.py의 run_stage(device)가 담당합니다.',
        code: compactCodeLines(source, [
          'print(json.dumps(run_stage(device), indent=2, ensure_ascii=False))',
          "if __name__ == '__main__':",
          'main()',
        ]),
      },
    ],
  };
}

function coreCodeGuideSummaryFor(path, unit = null) {
  if (path.endsWith('analysis.py')) {
    return unit
      ? `${unit.title}의 실습 산출물을 다시 읽어, 빠진 metrics를 확인하고 핵심 숫자를 해석 문장·리포트로 바꾸는 흐름만 먼저 모았습니다.`
      : '해석 코드는 새 모델을 학습하는 파일이 아니라, 앞선 실행 산출물을 검증하고 지표를 사람이 읽을 수 있는 결론으로 바꾸는 파일입니다.';
  }
  return unit
    ? `${unit.title}의 목표(${stripMarkdownLinks(unit.objective || '단원 핵심')})와 직접 연결되는 코드만 먼저 모았습니다.`
    : '전체 코드를 한 번에 읽기 어렵다면, 아래 핵심 발췌에서 데이터가 들어와 계산·학습·평가·저장으로 이어지는 흐름만 먼저 잡고 전체 코드로 내려가면 됩니다.';
}

function coreCodeGuideTitleFor(path, lineCount = 0) {
  if (path.endsWith('scratch_lab.py')) return lineCount >= 130 ? '긴 기초 실습은 핵심 계산만 먼저 훑어보세요' : '기초 실습은 단원 핵심 계산만 먼저 훑어보세요';
  if (path.endsWith('framework_lab.py')) return lineCount >= 130 ? '긴 프레임워크 실습은 데이터→모델→평가 흐름만 먼저 보세요' : '프레임워크 실습은 단원 핵심 흐름만 먼저 보세요';
  if (path.endsWith('analysis.py')) return lineCount >= 130 ? '긴 해석 코드는 결과 읽기→검증→리포트 저장만 먼저 보세요' : '해석 코드는 결과 읽기→검증→저장만 먼저 보세요';
  if (path.endsWith('experiment.py')) return '긴 실험 흐름 코드는 실행 단계만 먼저 따라가세요';
  if (path.endsWith('run_stage.py')) return '긴 실행 코드는 옵션→실험 호출→결과 위치만 먼저 보세요';
  if (path.endsWith('dataset.py')) return '긴 데이터 코드는 입력 표를 만드는 흐름만 먼저 보세요';
  return '긴 파일은 핵심 흐름만 먼저 훑어보세요';
}

function unitCoreCategoryFor(unit, section) {
  if (!unit) return null;
  const terms = unitCoreTermsFor(unit);
  if (!terms.length) return null;
  const readableTerms = terms.slice(0, 3).join(', ');
  return {
    label: '단원 핵심 개념',
    note: `${unit.title}에서 먼저 잡아야 하는 ${readableTerms} 코드입니다. 일반 저장/시각화 보조 코드보다 이 부분을 먼저 보세요.`,
    terms,
    patterns: terms.map((term) => flexibleTermPattern(term)),
  };
}

function unitCoreTermsFor(unit) {
  const sourceTerms = [
    ...(unit.key_terms || []),
    unit.objective || '',
    ...(unit.analysis_questions || []),
  ];
  const terms = [];
  const add = (value) => {
    const text = String(value || '').trim();
    if (!text) return;
    text
      .split(/[^A-Za-z0-9_]+/)
      .filter((part) => part.length >= 3 || /^l2$/i.test(part))
      .forEach((part) => terms.push(part));
  };
  sourceTerms.forEach(add);
  const haystack = sourceTerms.join(' ').toLowerCase();
  const aliasGroups = [
    [/normalization|normalise|normalize|z-score|layernorm|정규화/, ['normalization', 'normalize', 'normalized', 'zscore', 'z_score', 'LayerNorm', 'layer_norm']],
    [/regularization|regularisation|weight decay|l2|dropout|규제/, ['regularization', 'weight_decay', 'weight decay', 'l2', 'dropout']],
    [/gradient|backprop|미분|역전파/, ['gradient', 'grad', 'backward', 'finite_difference']],
    [/tensor|shape|broadcast|matmul|텐서/, ['shape', 'reshape', 'broadcast', 'matmul', 'softmax']],
    [/token|tokenization|embedding|토큰|임베딩/, ['tokenize', 'tokenizer', 'token', 'embedding', 'encode']],
    [/attention|transformer|어텐션/, ['attention', 'query', 'key', 'value', 'softmax']],
    [/retrieval|rag|검색/, ['retrieve', 'retrieval', 'rank', 'grounding']],
    [/reward|preference|rlhf|dpo|orpo|kto/, ['reward', 'preference', 'margin', 'policy']],
    [/distributed|torchrun|ddp|zero|fsdp|parallel|rank/, ['rank', 'world_size', 'all_reduce', 'stage', 'shard', 'parallel']],
    [/multimodal|vision|image|vlm|vla|action/, ['image', 'vision', 'action', 'alignment', 'embedding']],
  ];
  aliasGroups.forEach(([pattern, aliases]) => {
    if (pattern.test(haystack)) terms.push(...aliases);
  });
  return uniqueByNormalized(terms).slice(0, 18);
}

function uniqueByNormalized(values) {
  const seen = new Set();
  return values.filter((value) => {
    const key = String(value).toLowerCase().replace(/[^a-z0-9]/g, '');
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function flexibleTermPattern(term) {
  const normalized = String(term).trim();
  if (!normalized) return /a^/;
  const body = escapeRegExp(normalized).replace(/[\s_-]+/g, '[_\\s-]*');
  return new RegExp(body, 'i');
}

function automaticCoreCodeSteps(section, source, symbols, unit = null) {
  const path = cleanHref(section.href);
  if (path.endsWith('analysis.py')) return analysisCoreCodeSteps(section, source, symbols, unit);
  const steps = [];
  const used = new Set();
  const categories = [
    {
      label: '전체 실행 흐름',
      note: '이 파일의 진입점입니다. 입력 준비, 핵심 계산, metric 저장이 어떤 순서로 이어지는지 먼저 봅니다.',
      terms: ['run_stage', 'run', 'main'],
      patterns: [/^def\s+(run_stage|run|main)\s*\(/m],
    },
    {
      label: '입력과 설정 준비',
      note: '데이터, 예제, 설정값이 어떤 형태로 만들어져 뒤 계산으로 들어가는지 확인합니다.',
      terms: ['load', 'read', 'prepare', 'preprocess', 'preprocessor', 'build_dataset', 'dataset', 'make_', 'generate', 'tokenize', 'split', 'sample', 'batch', 'contract', 'context'],
      patterns: [/DataLoader|Dataset|train_test_split|build_dataset|load_|read_csv|EXAMPLES|SAMPLES|CONFIG|toy_batch|build_toy|make_split/],
    },
    {
      label: '모델·변환·핵심 계산',
      note: '입력이 logit, score, embedding, action, reconstruction 같은 중간 결과로 바뀌는 핵심 계산입니다.',
      terms: ['forward', 'model', 'encode', 'decode', 'attention', 'logit', 'score', 'compute', 'simulate', 'retrieve', 'rank'],
      patterns: [/model\s*=|logits?\s*=|scores?\s*=|softmax|forward\(|encode|decode|attention|retrieve|rank/],
    },
    {
      label: 'loss·metric·판정 기준',
      note: '실험이 잘 됐는지 판단하는 loss, reward, accuracy, F1, ranking metric 같은 기준입니다.',
      terms: ['loss', 'metric', 'accuracy', 'f1', 'reward', 'evaluate', 'score', 'select', 'threshold', 'coverage'],
      patterns: [/loss|accuracy|f1|reward|metric|threshold|evaluate|roc_auc|mean_squared_error|precision_recall_curve|confusion_matrix/],
    },
    {
      label: '학습·업데이트·반복 루프',
      note: '파라미터나 후보가 반복적으로 바뀌는 구간입니다. optimizer, epoch, step, update를 먼저 찾으세요.',
      terms: ['train', 'step', 'update', 'adapt', 'optimize', 'epoch', 'fit', 'accumulate'],
      patterns: [/for\s+epoch|optimizer|backward\(|\.step\(|\.fit\(|train_recipe|train_epoch|def\s+train_|update|adapt/],
    },
    {
      label: '결과 저장과 리포트',
      note: '브라우저에서 다시 볼 지표, 그림, 해석 노트가 어디에 저장되는지 확인합니다.',
      terms: ['run', 'save', 'write', 'render', 'report', 'svg', 'main'],
      patterns: [/METRICS_PATH|FIGURE_PATH|write_text|json\.dumps|to_csv|savefig|summary|report|artifact/i],
    },
  ].filter(Boolean);

  categories.forEach((category) => {
    const step = coreStepFromFunctionCategory(section, source, symbols, category, used)
      || coreStepFromPatternCategory(source, category, used);
    if (step) steps.push(step);
  });

  if (!steps.length && symbols.length) {
    symbols.slice(0, 4).forEach((name) => {
      const code = extractFunctionExcerpt(source, name);
      if (!code || used.has(code)) return;
      used.add(code);
      steps.push({
        label: `${name}() 먼저 보기`,
        note: roleHintForFunction(name, section) || '단원 실행 흐름에서 어떤 입력을 받아 어떤 결과로 바꾸는지 확인하세요.',
        code,
      });
    });
  }

  return steps;
}

function analysisCoreCodeSteps(section, source, symbols, unit = null) {
  const steps = [];
  const used = new Set();
  const categories = [
    {
      label: '필수 산출물 확인',
      note: 'analysis.py는 앞선 실습 결과가 있어야 의미가 있습니다. 누락된 metrics나 report를 먼저 확인하는 방어 코드부터 보세요.',
      terms: ['ensure', 'missing', 'required', 'exists'],
      patterns: [/_ensure|missing\s*=|\.exists\(\)|raise\s+SystemExit|FileNotFoundError/],
    },
    {
      label: '실습 결과 읽기',
      note: 'scratch/framework/experiment가 만든 JSON·표·리포트를 해석 입력으로 다시 읽는 구간입니다.',
      terms: ['load', 'read', 'json'],
      patterns: [/_load_json|json\.loads|json\.load|read_text|pd\.read_|metrics\s*=/],
    },
    {
      label: '핵심 지표 비교',
      note: unit
        ? `${unit.title}의 결론을 만들 숫자입니다. ratio, delta, loss, accuracy 같은 비교값이 어떤 원자료에서 나왔는지 확인하세요.`
        : 'ratio, delta, loss, accuracy 같은 비교값이 어떤 원자료에서 나왔는지 확인하세요.',
      terms: ['ratio', 'delta', 'loss', 'accuracy', 'f1', 'score', 'metric'],
      patterns: [/ratio\s*=|delta\s*=|accuracy\s*=|f1\s*=|score\s*=|loss\s*=|metric/],
    },
    {
      label: '해석 문장 만들기',
      note: '숫자를 그대로 나열하지 않고, 학습자가 가져가야 할 결론·주의점·다음 질문으로 바꾸는 구간입니다.',
      terms: ['comment', 'summary', 'interpret', 'report'],
      patterns: [/comment\s*=|summary\s*=|interpret|observed_report\s*=|report\s*=/],
    },
    {
      label: '해석 리포트 저장',
      note: '사이트와 다음 학습 단계에서 다시 열어볼 analysis/report 파일을 쓰는 마지막 구간입니다.',
      terms: ['write', 'save', 'report'],
      patterns: [/write_text|json\.dumps|to_markdown|to_csv|savefig/],
    },
  ];

  categories.forEach((category) => {
    const step = coreStepFromFunctionCategory(section, source, symbols, category, used)
      || coreStepFromPatternCategory(source, category, used);
    if (step) steps.push(step);
  });

  if (!steps.length && symbols.length) {
    symbols.slice(0, 4).forEach((name) => {
      const code = extractFunctionExcerpt(source, name);
      if (!code || used.has(code)) return;
      used.add(code);
      steps.push({
        label: `${name}() 먼저 보기`,
        note: roleHintForFunction(name, section) || '해석 코드에서 어떤 결과물을 읽어 어떤 문장이나 리포트로 바꾸는지 확인하세요.',
        code,
      });
    });
  }

  return steps.slice(0, 4);
}

function coreStepFromFunctionCategory(section, source, symbols, category, used) {
  const name = symbols.find((symbol) => category.terms.some((term) => symbol.toLowerCase().includes(term)) && !used.has(`fn:${symbol}`));
  if (!name) return null;
  const code = extractFunctionExcerpt(source, name);
  if (!code || used.has(code)) return null;
  used.add(`fn:${name}`);
  used.add(code);
  const presentation = coreFunctionStepPresentation(name, code, category, section);
  return {
    label: presentation.label,
    note: presentation.note,
    code,
  };
}

function coreFunctionStepPresentation(name, code, category, section) {
  const baseName = `${name}()`;
  const normalizedName = name.toLowerCase();
  const lower = `${name}\n${code}`.toLowerCase();
  const roleHint = roleHintForFunction(name, section);
  if (/run_stage/.test(normalizedName)) {
    return { label: `실험 전체를 묶는 ${baseName}`, note: '데이터 split, 모델 비교, best model 선택, 지표/그림 저장이 한 번에 연결되는 stage 실행 진입점입니다.' };
  }
  if (/load|read|dataset|split|preprocess|preprocessor|toy_batch|build_.*input|make_split/.test(normalizedName)) {
    return { label: `입력 데이터 준비: ${baseName}`, note: roleHint || '원본 데이터나 toy batch를 모델이 읽을 수 있는 feature, target, mask, split 형태로 만드는 코드입니다.' };
  }
  if (/rank|retriev/.test(normalizedName)) {
    return { label: `검색 순위 계산: ${baseName}`, note: 'query와 후보의 점수를 정렬해 top-k와 retrieval metric의 입력을 만듭니다.' };
  }
  if (/metric|accuracy|f1|recall|mrr|ndcg|score|coverage|loss/.test(normalizedName)) {
    return { label: `판단 지표 계산: ${baseName}`, note: roleHint || '실행 결과가 좋았는지 판단하는 숫자를 만드는 코드입니다.' };
  }
  if (/token|encode|embedding|mask|batch/.test(normalizedName)) {
    return { label: `입력 표현 만들기: ${baseName}`, note: roleHint || '원본 예제를 token, tensor, batch, mask처럼 모델이 읽을 수 있는 표현으로 바꿉니다.' };
  }
  if (/train|fit|adapt|optimizer|step/.test(normalizedName) || /optimizer|backward|loss_history/.test(lower)) {
    return { label: `학습 루프: ${baseName}`, note: roleHint || 'loss 계산, backward, optimizer step이 실제로 파라미터를 바꾸는 흐름입니다.' };
  }
  if (/write|save|report|main/.test(normalizedName)) {
    return { label: `산출물 저장 흐름: ${baseName}`, note: roleHint || 'metrics, figure, report처럼 사이트에서 다시 확인할 결과를 남기는 부분입니다.' };
  }
  return { label: `${category.label}: ${baseName}`, note: roleHint || category.note };
}

function coreStepFromPatternCategory(source, category, used) {
  const code = extractPatternExcerpt(source, category.patterns);
  if (!code || used.has(code)) return null;
  used.add(code);
  const presentation = coreStepPresentationFromCode(code, category);
  return {
    label: presentation.label,
    note: presentation.note,
    code,
  };
}

function coreStepPresentationFromCode(code, category) {
  const text = String(code || '');
  const lower = text.toLowerCase();
  const match = (pattern) => pattern.test(lower) || pattern.test(text);
  if (match(/action_head|safety_head|train_policy|safety_loss|action_loss/)) {
    return {
      label: 'action과 safety head가 함께 학습되는 지점',
      note: 'VLA에서는 action accuracy와 safety gate를 분리해서 봐야 하므로 두 head와 두 loss가 만나는 코드를 먼저 확인합니다.',
    };
  }
  if (match(/rank|retrieval|recall_at|mrr|ndcg|top_k|topk/)) {
    return {
      label: 'ranking과 retrieval metric을 만드는 지점',
      note: 'query-document 점수에서 top-k 순위와 recall/MRR/NDCG 같은 판단 기준이 만들어지는 흐름입니다.',
    };
  }
  if (match(/token|embedding|padding|mask|vocab/)) {
    return {
      label: 'token id·embedding·mask가 연결되는 지점',
      note: '문장이 token id로 바뀌고 embedding/mask를 거쳐 모델 입력 shape가 되는 흐름을 봅니다.',
    };
  }
  if (match(/schedule|microbatch|bubble|pipeline_stage|stage_partition|partition_boundary/)) {
    return {
      label: 'stage schedule과 bubble을 계산하는 지점',
      note: 'pipeline parallelism은 레이어 분할뿐 아니라 microbatch 시간표와 bubble/throughput 해석이 핵심입니다.',
    };
  }
  if (match(/shard|all_gather|reduce_scatter|zero|fsdp|tensor_parallel|collective/)) {
    return {
      label: 'shard와 collective trade-off가 드러나는 지점',
      note: '분산 학습 코드는 무엇을 나누고 언제 다시 모으는지, 그때 memory/communication이 어떻게 바뀌는지 봐야 합니다.',
    };
  }
  if (match(/optimizer|backward|\.step\(|loss\s*=|cross_entropy|binary_cross_entropy|mse_loss/)) {
    return {
      label: 'loss에서 optimizer step으로 이어지는 지점',
      note: '모델 출력이 loss가 되고, backward/step을 통해 파라미터가 바뀌는 학습 루프의 중심입니다.',
    };
  }
  if (match(/metrics\s*=|metrics_path|write_text|json\.dumps|to_csv|savefig|artifact|report/)) {
    return {
      label: '지표와 리포트를 저장하는 지점',
      note: '사이트와 해석 노트에서 다시 볼 숫자·그림·표가 어느 이름으로 저장되는지 확인합니다.',
    };
  }
  if (match(/model\s*=|nn\.|pipeline\(|classifier|regressor|forward\(/)) {
    return {
      label: '모델 또는 변환이 정의되는 지점',
      note: '입력 feature가 logit, score, prediction, embedding 같은 비교 가능한 출력으로 바뀌는 코드입니다.',
    };
  }
  return { label: category.label, note: category.note };
}

function compactCodeLines(source, preferredLines) {
  const sourceLines = source.split('\n').map((line) => line.trim());
  return preferredLines
    .map((preferred) => {
      const exact = sourceLines.find((line) => line === preferred);
      if (exact) return exact;
      if (preferred.includes('loss_plus = forward_loss(weight + epsilon, bias)')) return '_, loss_plus = forward_loss(weight + epsilon, bias)';
      if (preferred.includes('loss_minus = forward_loss(weight - epsilon, bias)')) return '_, loss_minus = forward_loss(weight - epsilon, bias)';
      return preferred;
    })
    .join('\n');
}

function extractFunctionSignature(source, name) {
  const match = source.match(new RegExp(`^def\\s+${escapeRegExp(name)}\\s*\\([^\\n]*\\):`, 'm'));
  return match ? match[0] : `${name}(...)`;
}

function extractFunctionExcerpt(source, name) {
  const lines = source.split('\n');
  const start = lines.findIndex((line) => new RegExp(`^def\\s+${escapeRegExp(name)}\\s*\\(`).test(line));
  if (start < 0) return extractFunctionSignature(source, name);
  const snippet = [lines[start]];
  for (let index = start + 1; index < lines.length && snippet.length < 5; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    if (/^(def|class)\s+/.test(line)) break;
    if (!trimmed || trimmed.startsWith('#')) continue;
    if (trimmed.startsWith('"""') || trimmed.startsWith("'''")) continue;
    snippet.push(line);
  }
  if (snippet.length === 1) return extractFunctionSignature(source, name);
  if (!/^(def|class)\s+/.test(lines[start + snippet.length] || '')) snippet.push('    ...');
  return trimCommonIndent(snippet).join('\n');
}

function extractPatternExcerpt(source, patterns) {
  const lines = source.split('\n');
  const index = lines.findIndex((line) => patterns.some((pattern) => pattern.test(line)));
  if (index < 0) return '';
  const start = Math.max(0, index - 2);
  const end = Math.min(lines.length, index + 3);
  return trimCommonIndent(lines.slice(start, end).filter((line) => line.trim())).join('\n');
}

function trimCommonIndent(lines) {
  const indents = lines
    .filter((line) => line.trim())
    .map((line) => (line.match(/^\s*/) || [''])[0].length);
  const minIndent = indents.length ? Math.min(...indents) : 0;
  return lines.map((line) => line.slice(minIndent).trimEnd());
}

function annotateCodeWithInlineHints(section, source) {
  return annotateFunctionRoleHints(annotateArtifactLocations(source, section), section);
}

function annotateArtifactLocations(source, section) {
  const artifactHint = '# 학습 포인트: 이 경로가 실행 후 지표/그림/리포트가 남는 위치입니다.';
  const reportHint = '# 학습 포인트: analysis.py가 최종 해석 문서를 쓰는 위치입니다.';
  let annotated = source;
  if (/^ARTIFACT_DIR\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^ARTIFACT_DIR\s*=)/m, `${artifactHint}\n$1`);
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^REPORT\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^REPORT\s*=)/m, `${reportHint}\n$1`);
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^SCRATCH\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^SCRATCH\s*=)/m, '# 학습 포인트: 기초/프레임워크 지표를 해석 입력으로 다시 읽습니다.\n$1');
  }
  if (cleanHref(section.href).endsWith('analysis.py') && /^ANALYSIS_PATH\s*=/m.test(annotated)) {
    annotated = annotated.replace(/(^ANALYSIS_PATH\s*=)/m, '# 학습 포인트: 분석 결과 문서가 저장되는 위치입니다.\n$1');
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
  if (normalized.includes('ensure') || normalized.includes('exists') || normalized.includes('missing')) return '앞선 실행 산출물이 빠졌는지 확인하고, 없으면 무엇을 먼저 실행해야 하는지 알려주는 방어 코드입니다.';
  if (normalized.includes('forward')) return 'tensor 입력이 logit·embedding·action 같은 모델 출력으로 바뀌는 계산 경로입니다.';
  if (normalized.includes('train')) return 'batch → loss → optimizer step이 연결되는 학습 루프입니다.';
  if (normalized.includes('evaluate') || normalized.includes('metric') || normalized.includes('score')) return '단원에서 비교할 지표를 계산하므로 단원 안내의 성공 기준과 나란히 확인하세요.';
  if (normalized.includes('compute') || normalized.includes('calculate')) return '중간 텐서나 수치를 최종 지표로 바꾸는 계산입니다.';
  if (normalized.includes('build') || normalized.includes('create') || normalized.includes('prepare') || normalized.includes('make')) return '작은 데이터, 모델, 설정 중 무엇을 고정해 비교 조건을 만드는지 확인하세요.';
  if (normalized.includes('generate') || normalized.includes('sample') || normalized.includes('decode')) return '모델 출력이 사람이 읽을 수 있는 토큰·설명·행동으로 바뀌는 지점입니다.';
  if (normalized.includes('write') || normalized.includes('save')) return '브라우저와 해석 노트가 다시 볼 결과물을 저장하는 지점입니다.';
  if ((normalized.includes('load') || normalized.includes('read')) && cleanHref(section.href).endsWith('dataset.py')) return '원본 표 데이터를 읽어 실험 입력으로 만드는 지점입니다.';
  if (normalized.includes('load') || normalized.includes('read')) return '이전 실행 산출물을 다시 읽어 분석 입력으로 바꾸는 지점입니다.';
  return '';
}

function sectionSpecificRunHint(section) {
  const path = cleanHref(section.href);
  if (path.endsWith('scratch_lab.py')) return '작은 입력 예제를 만들고 직접 계산한 뒤 지표와 그림으로 남기는 흐름입니다.';
  if (path.endsWith('framework_lab.py')) return '프레임워크 모델·학습/평가 설정을 묶어 기초 실습 결과와 비교할 지표를 만듭니다.';
  if (path.endsWith('analysis.py')) return '앞선 두 실습의 결과물을 읽고 빠진 부분을 확인한 뒤 해석 노트나 요약을 작성합니다.';
  if (path.endsWith('run_stage.py')) return '실행 옵션을 받아 실험 전체를 한 번에 시작하는 연결부입니다.';
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
      title: '기초 실습 코드: 작은 숫자로 원리 확인하기',
      summary: '라이브러리 편의 기능보다 먼저, 눈으로 따라갈 수 있는 작은 계산으로 단원 핵심을 확인합니다.',
      what: '작은 예제 데이터를 만들고 계산 과정을 단계별로 저장해, 이론 설명이 어떤 숫자와 그림으로 드러나는지 보여줍니다.',
      howToRead: '위쪽의 데이터/설정 → 중간의 계산 함수 → 아래쪽의 결과 저장 순서로 읽으면 됩니다. 먼저 입력 모양과 중간 변수 이름을 보고, 마지막에 저장되는 지표를 확인하세요.',
      outputs: '보통 기초 실습 지표, 작은 그림/표, 실행 요약이 남습니다. 이 결과는 뒤의 해석 단계에서 비교 기준이 됩니다.',
      functions,
    };
  }
  if (path.endsWith('framework_lab.py')) {
    return {
      title: '프레임워크 실습 코드: 실제 도구로 같은 아이디어 확인하기',
      summary: '기초 실습에서 본 계산을 PyTorch나 sklearn 같은 도구로 다시 실행해, 손계산 감각과 실제 도구의 결과를 비교합니다.',
      what: '단원의 핵심 개념을 프레임워크로 구현해 학습 흐름, 평가 지표, 저장 형식이 어떻게 이어지는지 확인합니다.',
      howToRead: '데이터 준비 → 모델/파이프라인 정의 → 학습 또는 추론 → 결과 저장 순서로 따라가세요. 기초 실습과 같은 지표가 어떻게 대응되는지 비교하면 좋습니다.',
      outputs: '보통 프레임워크 지표, 결과 그림, 예측 샘플이 남습니다. 기초 실습 결과와 나란히 보며 도구가 자동으로 처리한 부분을 찾습니다.',
      functions,
    };
  }
  if (path.endsWith('analysis.py')) {
    return {
      title: '결과 해석 코드: 실행 결과를 공부 노트로 바꾸기',
      summary: '앞선 두 실습이 만든 지표와 결과물을 읽고, 무엇이 잘 됐고 어디서 실패했는지 한국어 분석 노트로 정리합니다.',
      what: '실험 결과물을 검증하고, 핵심 수치·실패 사례·다음 질문을 해석 노트나 요약으로 정리하는 단계입니다.',
      howToRead: '입력 결과를 읽는 부분 → 지표 검증/집계 → 설명 문장 생성 → 저장 경로 순서로 읽으세요. 오류 메시지는 어떤 결과물이 빠졌는지 알려주는 체크리스트 역할을 합니다.',
      outputs: '해석 노트, 요약, 관찰 지표 같은 결과물이 남습니다. 단원을 완료할 때는 이 노트의 질문에 답할 수 있어야 합니다.',
      functions,
    };
  }
  if (path.endsWith('dataset.py')) {
    return {
      title: '데이터 준비 코드: 실험에 넣을 표 만들기',
      summary: '원본 데이터를 읽고, 입력 표와 정답 열이 어떤 기준으로 나뉘는지 확인하는 출발점입니다.',
      what: '이 단계는 데이터 불러오기, 학습/평가 나누기, 입력/정답 구성, 결측·범주형 처리 준비를 담당합니다.',
      howToRead: '데이터를 불러오는 함수 → 입력 열 선택 → 정답 생성 → 나누기/전처리 입력 형태 순서로 읽으세요. 마지막에 다음 실험 단계가 기대하는 반환 모양을 확인합니다.',
      outputs: '대개 직접 결과 파일을 저장하기보다, 다음 실험 단계가 학습과 평가에 사용할 입력 묶음을 넘깁니다.',
      functions,
    };
  }
  if (path.endsWith('experiment.py')) {
    return {
      title: '실험 흐름 코드: 준비·학습·평가 연결하기',
      summary: '기준 모델, 전처리, 모델 학습, 지표 계산, 결과 저장이 한곳에서 연결되므로 이 단원의 핵심 흐름입니다.',
      what: '데이터 준비 단계가 만든 입력을 받아 모델을 학습·비교하고, 지표·그림·예측 샘플을 저장하는 연결 코드입니다.',
      howToRead: '설정값 → 데이터 준비 호출 → 기준 모델/모델 정의 → 학습/예측 → 지표 저장 순서로 읽으세요. 실험 실행 코드는 보통 이 흐름을 한 번에 호출합니다.',
      outputs: '실험 설정, 지표, 예측 샘플, 그림이 남고 리포트/해석 단계가 이를 읽어 결론을 만듭니다.',
      functions,
    };
  }
  if (path.endsWith('run_stage.py')) {
    return {
      title: '실험 실행 코드: 한 번에 실행하고 결과 모으기',
      summary: '데이터 준비와 실험 흐름에 흩어진 준비·학습·평가 과정을 한 번에 재현 가능하게 묶습니다.',
      what: '이 단계는 단원 실험을 실행하고, 선택된 CPU/GPU 환경에서 지표와 그림을 결과 폴더에 남기도록 연결합니다.',
      howToRead: '실행 옵션 확인 → 재실행 기준값/장치 설정 → 실험 흐름 호출 → 요약 출력 순서로 읽으세요. 실제 모델 비교와 저장 로직은 실험 흐름 코드에서 이어서 확인합니다.',
      outputs: '결과 폴더에 지표, 설정, 예측 샘플, 그림과 실행 요약이 남습니다. 실패하면 실행 환경이나 의존성 확인이 필요한 지점입니다.',
      functions,
    };
  }
  return {
    title: `${displaySectionLabel(section)} 설명`,
    summary: '이 코드는 단원 실습을 재현 가능하게 실행하기 위한 보조 코드입니다.',
    what: '자료 탭과 단원 안내의 실행 순서를 함께 보며 역할을 확인하세요.',
    howToRead: '상단 설정, 데이터 준비, 계산/호출 지점, 저장 로직 순서로 읽으면 됩니다.',
    outputs: '실행 요약, 지표 파일, 해석 노트, 그림 중 하나 이상의 결과물을 남깁니다.',
    functions,
  };
}

function extractPythonSymbols(source) {
  return extractPythonSymbolNames(source).map((name) => `${name}()`).slice(0, 8);
}

function extractPythonSymbolNames(source) {
  return Array.from(source.matchAll(/^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/gm))
    .map((match) => match[1]);
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
  const label = checkpointDisplayLabel(section.checkpoint);
  button.textContent = checked[section.checkpoint] ? '읽음 표시됨' : `${label} 완료로 표시`;
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
      const displayLabel = friendlyDocumentLabel(label, clean);
      return `<button type="button" class="inline-doc-link" data-doc-href="${escapeHtml(resolved)}" data-doc-label="${escapeHtml(displayLabel)}">${displayLabel}</button>`;
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
  const llmFastPath = [
    '00_foundations/01_tensor_shapes',
    '00_foundations/02_activation_and_loss',
    '00_foundations/03_gradients_and_backpropagation',
    '01_ml/01_tabular_classification',
    '01_ml/03_model_selection_and_interpretation',
    '02_deep_learning/01_perceptron_and_mlp',
    '02_deep_learning/03_sequence_models_rnn_lstm_gru',
    '02_deep_learning/04_attention_and_transformers',
    '02_deep_learning/07_training_recipes_and_debugging',
    '03_nlp_bridge/01_tokenization_and_embeddings',
    '03_nlp_bridge/02_attention_and_transformer_block',
    '04_nlp/01_text_classification',
    '05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives',
    '05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture',
    '05_advanced_nlp_llm/04_instruction_tuning_and_sft',
    '05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto',
    '05_advanced_nlp_llm/06_rlhf_and_reasoning_rl',
  ];
  const multimodalVlaPath = [
    ...llmFastPath,
    '08_multimodal_bridge/01_contrastive_alignment',
    '09_multimodal/01_image_text_retrieval',
    '09_multimodal/02_image_captioning',
    '09_multimodal/03_visual_question_answering',
    '10_vla/01_vision_language_action_grounding',
  ];
  const systemsPath = [
    '00_foundations/01_tensor_shapes',
    '00_foundations/05_gpu_memory_runtime',
    '02_deep_learning/07_training_recipes_and_debugging',
    '06_training_systems/01_torchrun_and_ddp_basics',
    '06_training_systems/02_accelerate_workflows',
    '06_training_systems/03_deepspeed_zero',
    '06_training_systems/04_fsdp_checkpointing_and_offload',
    '06_training_systems/05_tensor_parallelism',
    '06_training_systems/06_pipeline_parallelism',
    '06_training_systems/07_data_parallel_grad_accumulation',
    '06_training_systems/08_hybrid_parallel_topologies',
    '06_training_systems/09_profiling_monitoring_and_failure_recovery',
    '07_frontier_labs/01_paper_reproduction_playground',
    '07_frontier_labs/02_capstone_model_building',
    '07_frontier_labs/03_agentic_training_and_eval_loops',
    '07_frontier_labs/04_benchmark_and_dataset_construction',
    '07_frontier_labs/05_open_ended_research_tracks',
  ];
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
      description: '트랙 전체가 아니라 필수 단원만 압축해 LLM pretraining, SFT, preference/RLHF까지 먼저 도달합니다.',
      include: (unit) => llmFastPath.includes(unit.path),
    },
    {
      id: 'multimodal',
      label: 'Multimodal/VLA 경로',
      description: 'LLM 빠른 경로 위에 contrastive alignment, VQA, action-token grounding을 순서대로 붙입니다.',
      include: (unit) => multimodalVlaPath.includes(unit.path),
    },
    {
      id: 'systems',
      label: 'Systems 심화 경로',
      description: '모델 학습 감각 이후 distributed/system, frontier lab 실험 운영 능력을 강화합니다.',
      include: (unit) => systemsPath.includes(unit.path),
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
    { key: 'theory', label: '이론 읽기', description: '단원 안내와 핵심 이론으로 데이터셋, 기준 모델, 평가 지표의 역할을 먼저 잡는다.' },
    { key: 'lab-setup', label: '실습 구성 살펴보기', description: '데이터 준비 코드와 실험 흐름 코드에서 데이터 생성, 모델, 평가 지표가 어떻게 연결되는지 본다.' },
    { key: 'run-stage', label: '실험 실행하기', description: '실험 실행 코드에서 어떤 단계가 어떤 결과물을 만드는지 확인한다.' },
    { key: 'analysis', label: '결과 정리하기', description: '결과 해석 코드와 리포트 코드로 결과를 해석하고 다음 실험 질문을 남긴다.' },
  ] : [
    { key: 'theory', label: '이론 읽기', description: '단원 안내, 핵심 이론, 준비 확인으로 왜 배우는지와 선행 개념을 잡는다.' },
    { key: 'scratch', label: '기초 실습하기', description: '기초 실습 코드로 작은 수치와 직접 계산을 확인한다.' },
    { key: 'framework', label: '도구로 다시 확인하기', description: '프레임워크 실습 코드로 PyTorch나 sklearn 관측을 비교한다.' },
    { key: 'analysis', label: '결과 정리하기', description: '결과 해석 코드와 해석 노트로 관측값을 한국어 설명으로 남긴다.' },
    { key: 'reflection', label: '회고 남기기', description: '회고 메모에 헷갈린 점, 실패 사례, 다음 질문을 적는다.' },
  ];
  return steps.filter((step) => {
    if (step.key === 'scratch') return unit.checkpoints.includes('scratch lab');
    if (step.key === 'framework') return unit.checkpoints.includes('framework lab');
    if (step.key === 'lab-setup') return unit.checkpoints.includes('실습 구성');
    if (step.key === 'run-stage') return unit.checkpoints.includes('실행 명령');
    if (step.key === 'analysis') return unit.checkpoints.includes('analysis script') || unit.checkpoints.includes('analysis note');
    if (step.key === 'reflection') return unit.checkpoints.includes('reflection');
    return true;
  });
}

function renderLessonGuidePlan(unit) {
  if (isIntroLesson(unit)) {
    return `<div class="start-callout">첫 단원에서는 사이트 사용법까지 같이 익힙니다. 읽기 → 실행 → 관찰 → 해석 → 메모로 이어지는 학습 루프를 여기서 한 번만 연습하세요.</div>
      <h3>처음 학습 순서</h3>
      <ol class="learning-steps">
        ${learningStepsFor(unit).map((step) => `<li><strong>${escapeHtml(step.label)}</strong><span>${escapeHtml(step.description)}</span></li>`).join('')}
      </ol>`;
  }
  return `<div class="start-callout">이제 공통 순서를 다시 외우기보다, 이 단원에서 무엇을 비교하고 어떤 증거를 남길지 먼저 정하세요.</div>
    <h3>이번 단원 브리핑</h3>
    <ol class="learning-steps focus-steps">
      ${lessonFocusStepsFor(unit).map((step) => `<li><strong>${escapeHtml(step.label)}</strong><span>${escapeHtml(step.description)}</span></li>`).join('')}
    </ol>`;
}

function isIntroLesson(unit) {
  return unit?.path === '00_foundations/01_tensor_shapes';
}

function lessonFocusStepsFor(unit) {
  const terms = unit.key_terms || [];
  const outputs = displayOutputList(unit.required_outputs || []);
  const questions = unit.analysis_questions || [];
  const primaryTerms = terms.slice(0, 3).join(', ') || '핵심 개념';
  const evidence = outputs.slice(0, 2).join(', ') || '지표와 해석 노트';
  const firstQuestion = questions[0] || '이번 실행 결과가 다음 단원과 어떻게 이어지는가?';
  const trap = questions[1] || unit.prereqs?.[0] || firstQuestion;
  const hasMlRunner = (unit.resources || []).some((resource) => resource.label === 'run_stage.py');
  return [
    {
      label: '지난 단원과 달라진 점',
      description: unit.objective || `${primaryTerms}를 이전 단원의 실행 감각과 연결하세요.`,
    },
    {
      label: '이번에 꼭 볼 것',
      description: `${primaryTerms}를 먼저 표시해 두고, 코드와 결과에서 이 용어들이 어디에 나타나는지 찾으세요.`,
    },
    {
      label: hasMlRunner ? '실험을 증명하는 산출물' : '이해를 증명하는 산출물',
      description: `${evidence}를 확인하고, 숫자나 그림 하나를 골라 “왜 이렇게 나왔는지” 설명하세요.`,
    },
    {
      label: '자주 틀리는 지점',
      description: `${trap} 이 질문에 답하지 못하면 결과를 봐도 이해가 남지 않습니다.`,
    },
  ];
}

function nextActionFor(unit, progress, selfCheckStats, quizItems, quizAnswers, checkpoints, checked) {
  const firstCode = lessonSectionsFor(unit).find((section) => section.type === 'code');
  const answered = quizItems.filter((question) => quizAnswers?.[question.id]).length;
  const prereqs = prerequisiteUnitsFor(unit);
  const unmetPrereq = prereqs.find((item) => item.path && lessonState(item.path).state !== 'done');
  if (unmetPrereq && /^(05_advanced_nlp_llm|08_multimodal_bridge|09_multimodal|10_vla)\//.test(unit.path)) {
    return `<strong>다음 행동: 선행 복습 먼저</strong><span>${escapeHtml(unmetPrereq.label)}을(를) 확인하면 이 단원의 코드와 실패 사례가 덜 막힙니다.</span>`;
  }
  const nextCheckpoint = (checkpoints || []).find((checkpoint) => !checked?.[checkpoint]);
  let action = '단원 안내부터 읽기';
  let detail = '목표와 선행 개념을 먼저 잡으세요.';
  if (nextCheckpoint) {
    action = checkpointActionLabel(nextCheckpoint, firstCode);
    detail = checkpointActionDetail(nextCheckpoint, unit);
  }
  if (!nextCheckpoint && answered < quizItems.length && selfCheckStats.done > 0) {
    action = '단원 점검 퀴즈 풀기';
    detail = '틀린 문제는 오답노트에 자동 저장됩니다.';
  }
  if (selfCheckStats.done === selfCheckStats.total && answered === quizItems.length && progress.state !== 'done') {
    action = '마무리하고 다음 단원으로';
    detail = '자가 점검과 퀴즈를 끝냈다면 다음 추천 단원으로 넘어가세요.';
  }
  return `<strong>다음 행동: ${escapeHtml(action)}</strong><span>${escapeHtml(detail)}</span>`;
}

function checkpointActionLabel(checkpoint, firstCode) {
  const normalized = String(checkpoint || '').toLowerCase();
  if (normalized === 'readme') return '단원 안내 읽기';
  if (normalized === 'theory') return '핵심 이론으로 원리 확인';
  if (normalized === 'prereqs') return '준비 확인으로 선행 개념 점검';
  if (normalized.includes('scratch')) return `${firstCode ? displaySectionLabel(firstCode) : '기초 실습 코드'} 실행`;
  if (normalized.includes('framework')) return '프레임워크 실습 코드 실행';
  if (normalized.includes('analysis')) return '결과 해석 정리';
  if (normalized.includes('reflection')) return '회고 메모 작성';
  if (normalized.includes('실습 구성')) return '데이터 준비와 실험 흐름 읽기';
  if (normalized.includes('실행 명령')) return '실험 실행 코드 실행';
  return `${checkpoint} 진행`;
}

function checkpointActionDetail(checkpoint, unit) {
  const normalized = String(checkpoint || '').toLowerCase();
  if (['readme', 'theory', 'prereqs'].includes(normalized)) return `${unitFocusSentence(unit)} 코드를 돌리기 전에 이 기준으로 단원 안내와 이론을 읽으세요.`;
  if (normalized.includes('scratch') || normalized.includes('framework') || normalized.includes('실행 명령')) return '실행 후 산출물 뷰어에서 이번 실행이 만든 지표와 그림을 확인하세요.';
  if (normalized.includes('analysis')) return '숫자를 결론으로 바꾸고, 실패 사례와 다음 실험 질문을 남기세요.';
  if (normalized.includes('reflection')) return '헷갈린 개념, 오답 이유, 다음 단원에서 확인할 질문을 한 줄 이상 남기세요.';
  return '완료하지 않은 체크포인트를 하나씩 닫으면 다음 행동이 갱신됩니다.';
}

function unitFocusSentence(unit) {
  const terms = (unit?.key_terms || []).slice(0, 2).join(', ') || '핵심 개념';
  const question = unit?.analysis_questions?.[0] || unit?.objective || '실행 결과가 무엇을 증명하는지';
  return `${terms}를 보면서 “${question}”에 답할 준비를 하세요.`;
}

function prerequisiteReadinessFor(unit) {
  const advanced = /^(05_advanced_nlp_llm|08_multimodal_bridge|09_multimodal|10_vla)\//.test(unit.path);
  if (!advanced) return '';
  const prereqs = prerequisiteUnitsFor(unit);
  const rows = prereqs.map((item) => {
    if (item.href) {
      return `<li><span class="chip in_progress">읽기 권장</span><button type="button" class="inline-doc-link" data-prereq-href="${escapeHtml(item.href)}" data-prereq-label="${escapeHtml(item.label)}">${escapeHtml(item.label)}</button></li>`;
    }
    const state = lessonState(item.path).state;
    const ready = state === 'done';
    return `<li><span class="chip ${ready ? 'done' : 'in_progress'}">${ready ? '확인됨' : '복습 권장'}</span><button type="button" class="inline-doc-link" data-prereq-unit="${escapeHtml(item.path)}">${escapeHtml(item.label)}</button></li>`;
  }).join('');
  return `<section class="prereq-gate" aria-label="선행 준비도">
    <strong>선행 준비도</strong>
    <p>고급 단원은 막지 않고 열어두지만, 아래 개념이 약하면 먼저 복습하는 편이 좋습니다.</p>
    <ul>${rows}</ul>
  </section>`;
}

function prerequisiteUnitsFor(unit) {
  const base = [
    { path: '00_foundations/01_tensor_shapes', label: 'Tensor shape와 broadcasting' },
    { path: '02_deep_learning/04_attention_and_transformers', label: 'Attention/Transformer block' },
  ];
  if (unit.path === '05_advanced_nlp_llm/06_rlhf_and_reasoning_rl') {
    return [
      ...base,
      { path: '05_advanced_nlp_llm/04_instruction_tuning_and_sft', label: 'SFT가 초기 assistant policy를 만드는 흐름' },
      { path: '05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto', label: 'chosen/rejected pair와 preference objective' },
      { href: '../docs/05_rl_primer_for_rlhf.md', label: 'RLHF용 RL 용어 입문 문서' },
    ];
  }
  if (unit.path.startsWith('05_advanced_nlp_llm/')) {
    return [...base, { path: '03_nlp_bridge/01_tokenization_and_embeddings', label: 'Tokenization/embedding' }];
  }
  if (unit.path.startsWith('08_multimodal_bridge/')) {
    return [
      ...base,
      { path: '05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives', label: 'Language modeling objective' },
      { href: '../docs/07_multimodal_generation_bridge.md', label: '멀티모달 생성 bridge 문서' },
    ];
  }
  if (unit.path.startsWith('09_multimodal/')) {
    return [
      ...base,
      { path: '05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives', label: 'Language modeling objective' },
      { path: '08_multimodal_bridge/01_contrastive_alignment', label: '이미지-텍스트 표현 정렬 감각' },
      { href: '../docs/07_multimodal_generation_bridge.md', label: '멀티모달 생성 bridge 문서' },
    ];
  }
  if (unit.path.startsWith('10_vla/')) {
    return [
      ...base,
      { path: '08_multimodal_bridge/01_contrastive_alignment', label: '이미지-텍스트 표현 정렬 감각' },
      { path: '09_multimodal/03_visual_question_answering', label: 'VQA와 grounding failure' },
      { path: '05_advanced_nlp_llm/06_rlhf_and_reasoning_rl', label: 'reward/policy/rollout/KL 용어' },
      { href: '../docs/08_rl_to_vla_bridge.md', label: 'RL→VLA bridge: MDP, trajectory, behavior cloning, offline RL' },
    ];
  }
  return base;
}


const LESSON_QUIZ_BLUEPRINTS = {
  '00_foundations/01_tensor_shapes': {
    prompt: 'matmul shape mismatch를 찾을 때 가장 먼저 맞춰야 하는 축은 무엇인가요?',
    explanation: 'matmul은 batch 차원보다 먼저 왼쪽 텐서의 마지막 축과 오른쪽 텐서의 마지막에서 두 번째 축이 서로 맞아야 합니다.',
    options: [
      { label: '왼쪽 마지막 차원과 오른쪽 마지막에서 두 번째 차원이 같은지 확인한다.', correct: true, explain: '이 축이 내적되는 축이라 mismatch의 핵심 근거입니다.' },
      { label: 'batch 차원만 같으면 matmul 내부 축은 자동으로 맞춰진다.', correct: false, explain: 'batch broadcasting과 matmul의 내적 축 조건은 서로 다릅니다.' },
      { label: '출력 shape만 보면 입력 shape 문제를 역추적하지 않아도 된다.', correct: false, explain: '출력만 보면 어떤 입력 축이 틀렸는지 놓치기 쉽습니다.' },
    ],
    shortPrompt: 'shape mismatch를 입력 축 기준으로 한 문장으로 설명해 보세요.',
    expected: '왼쪽 마지막 축과 오른쪽 마지막-두번째 축이 맞아야 하고, batch 축은 별도로 broadcast되는지 확인한다.',
  },
  '00_foundations/02_activation_and_loss': {
    prompt: '극단적인 입력값에서 sigmoid가 학습 신호를 약하게 만들 수 있는 이유는 무엇인가요?',
    explanation: 'sigmoid는 큰 양수/음수에서 0이나 1 근처로 포화되어 기울기가 작아집니다. 그래서 logits와 안정적 loss 구현을 함께 봐야 합니다.',
    options: [
      { label: '출력이 0 또는 1 근처로 포화되면 기울기가 작아져 업데이트 신호가 약해진다.', correct: true, explain: '포화 영역의 작은 gradient가 핵심입니다.' },
      { label: 'sigmoid는 음수를 모두 0으로 잘라 ReLU와 같은 sparse 출력을 만든다.', correct: false, explain: '음수를 0으로 자르는 것은 ReLU의 성질입니다.' },
      { label: 'softmax를 쓰면 BCE와 cross entropy의 target 형식 차이가 사라진다.', correct: false, explain: '손실 함수마다 기대하는 target 형식은 여전히 다릅니다.' },
    ],
    shortPrompt: 'activation 출력과 loss 입력(logit/probability)을 구분해 설명해 보세요.',
    expected: 'activation은 표현을 바꾸고 loss는 오차를 scalar로 압축한다. 안정적 loss는 probability 대신 logits를 직접 받기도 한다.',
  },
  '00_foundations/03_gradients_and_backpropagation': {
    prompt: 'analytic gradient와 finite-difference gradient가 거의 같다는 것은 무엇을 뜻하나요?',
    explanation: '손미분 chain rule이 실제 loss 변화율을 제대로 따라간다는 검산입니다.',
    options: [
      { label: 'chain rule로 계산한 gradient가 작은 epsilon 근사와 맞아 구현/미분 방향을 신뢰할 수 있다.', correct: true, explain: '두 값의 차이가 작을수록 미분 흐름이 맞다는 근거가 됩니다.' },
      { label: 'finite difference가 맞으면 learning rate를 아무리 키워도 loss가 줄어든다.', correct: false, explain: 'gradient 검산과 안정적인 step size는 별도 문제입니다.' },
      { label: 'bias gradient는 항상 0이므로 weight gradient만 보면 된다.', correct: false, explain: 'bias도 prediction 경로를 통해 loss에 영향을 줍니다.' },
    ],
    shortPrompt: 'backpropagation에서 local gradient들이 어떻게 곱해지는지 설명해 보세요.',
    expected: 'loss에서 출력으로 가는 gradient를 각 연산의 local derivative와 곱해 weight/bias 쪽으로 전달한다.',
  },
  '00_foundations/04_regularization_and_normalization': {
    prompt: '정규화와 weight decay를 같이 볼 때 가장 중요한 구분은 무엇인가요?',
    explanation: 'normalization은 입력/표현의 scale을 맞추고, weight decay는 파라미터 크기와 업데이트 방향에 제약을 더합니다.',
    options: [
      { label: '정규화는 gradient scale을 안정화하고, weight decay는 loss와 별개로 weight norm을 억제한다.', correct: true, explain: '두 기법의 작동 지점이 다릅니다.' },
      { label: 'weight decay를 켜면 step 전 data loss 값 자체가 반드시 달라진다.', correct: false, explain: 'PyTorch SGD에서는 data loss는 같아 보이고 optimizer step에서 decay가 반영될 수 있습니다.' },
      { label: 'dropout은 평가 모드에서도 항상 일부 값을 0으로 만든다.', correct: false, explain: 'dropout은 train/eval mode에 따라 동작이 달라집니다.' },
    ],
    shortPrompt: 'normalization과 regularization의 작동 위치를 구분해 설명해 보세요.',
    expected: 'normalization은 입력/표현 scale을 맞추고 regularization은 weight 크기나 경로 의존도를 억제한다.',
  },
  '00_foundations/05_gpu_memory_runtime': {
    prompt: 'training에서 batch size가 커질 때 특히 같이 커지는 메모리 항목은 무엇인가요?',
    explanation: 'parameter는 모델 크기에 묶이지만 activation은 batch와 sequence/feature 크기에 민감합니다. training은 gradient와 optimizer state도 추가로 필요합니다.',
    options: [
      { label: 'activation 메모리는 batch와 함께 커지고, training은 gradient/optimizer state까지 더 필요하다.', correct: true, explain: 'training과 inference의 메모리 차이를 만드는 핵심입니다.' },
      { label: 'parameter 메모리는 batch size가 커질 때마다 같은 비율로 늘어난다.', correct: false, explain: 'parameter 수는 batch가 아니라 모델 구조에 의해 결정됩니다.' },
      { label: 'mixed precision은 항상 메모리를 0에 가깝게 만들어 OOM을 없앤다.', correct: false, explain: '도움은 되지만 activation/optimizer/state 병목은 여전히 남습니다.' },
    ],
    shortPrompt: 'training과 inference의 메모리 항목 차이를 설명해 보세요.',
    expected: 'inference는 주로 parameter와 activation이지만 training은 activation 저장, gradient, optimizer state가 추가된다.',
  },
  '01_ml/01_tabular_classification': {
    prompt: 'majority baseline과 학습 모델의 차이를 가장 정직하게 판단하려면 무엇을 봐야 하나요?',
    explanation: '분류는 단일 정확도보다 class imbalance, AUPRC/F1, confusion matrix, error slice를 같이 봐야 합니다.',
    options: [
      { label: 'primary metric과 confusion/error slice를 함께 보고 baseline 대비 개선이 특정 class에만 치우치지 않았는지 확인한다.', correct: true, explain: '지표와 오류 분포를 함께 보는 것이 supervised ML의 기본입니다.' },
      { label: 'accuracy가 높으면 class imbalance나 error slice는 보지 않아도 된다.', correct: false, explain: '불균형 데이터에서는 accuracy가 baseline 착시를 만들 수 있습니다.' },
      { label: '가장 복잡한 모델을 고르면 feature 처리 실패는 자동으로 사라진다.', correct: false, explain: '전처리와 slice 오류는 모델 복잡도와 별도로 확인해야 합니다.' },
    ],
    shortPrompt: 'baseline, primary metric, error slice를 연결해 모델 선택 근거를 적어 보세요.',
    expected: 'baseline 대비 primary metric 개선과 confusion/error slice의 실패 패턴을 함께 근거로 삼는다.',
  },
  '01_ml/02_tabular_regression': {
    prompt: 'MAE와 RMSE가 서로 다르게 움직이면 어떤 해석이 필요한가요?',
    explanation: 'RMSE는 큰 오차에 더 민감하므로 outlier나 특정 구간 residual을 따로 봐야 합니다.',
    options: [
      { label: '큰 residual/outlier가 RMSE를 밀어 올리는지 residual summary와 prediction-vs-target을 확인한다.', correct: true, explain: 'MAE/RMSE 차이는 오차 분포 모양의 단서입니다.' },
      { label: 'RMSE가 MAE보다 크면 모델이 항상 틀렸다는 뜻이다.', correct: false, explain: '스케일과 outlier 민감도 차이를 해석해야 합니다.' },
      { label: 'R2만 높으면 residual 분포는 볼 필요가 없다.', correct: false, explain: 'R2가 좋아도 특정 구간 bias가 남을 수 있습니다.' },
    ],
    shortPrompt: 'MAE, RMSE, residual을 함께 사용해 회귀 모델을 평가해 보세요.',
    expected: '평균 오차, 큰 오차 민감도, residual 패턴을 함께 보며 baseline 대비 개선을 판단한다.',
  },
  '01_ml/03_model_selection_and_interpretation': {
    prompt: 'validation score가 가장 높은 모델을 바로 선택하면 위험한 이유는 무엇인가요?',
    explanation: '검증 점수는 leakage, spurious correlation, 해석 가능성, 운영 비용과 함께 봐야 합니다.',
    options: [
      { label: 'feature importance와 leakage 의심 신호를 함께 확인해 점수 상승이 믿을 수 있는지 검증해야 한다.', correct: true, explain: '성능과 해석 근거를 함께 남기는 것이 모델 선택입니다.' },
      { label: 'validation score가 높으면 leakage 가능성은 원천적으로 없다.', correct: false, explain: 'leakage는 오히려 비정상적으로 높은 score로 드러날 수 있습니다.' },
      { label: 'feature importance는 모델 선택과 무관한 시각화 장식이다.', correct: false, explain: '중요 특징은 해석과 오류 원인 추적의 핵심 근거입니다.' },
    ],
    shortPrompt: 'validation score와 feature importance를 함께 쓰는 이유를 설명해 보세요.',
    expected: 'score만이 아니라 leakage/spurious feature/운영 해석 가능성을 함께 검토하기 위해서다.',
  },
  '01_ml/04_large_scale_tabular': {
    prompt: '큰 표형 데이터에서 chunking을 쓰면 무엇을 반드시 같이 비교해야 하나요?',
    explanation: 'chunking은 메모리를 줄이지만 metric 일관성, ordering, throughput, reproducibility를 함께 관리해야 합니다.',
    options: [
      { label: '메모리/throughput 개선과 metric 재현성, 데이터 순서 보존을 함께 확인한다.', correct: true, explain: '큰 입력에서도 같은 실험 계약을 유지해야 합니다.' },
      { label: 'chunk를 쓰면 metric은 무조건 동일하므로 메모리만 보면 된다.', correct: false, explain: 'streaming 집계와 데이터 순서가 결과를 흔들 수 있습니다.' },
      { label: 'throughput이 높아지면 재현성 검사는 생략해도 된다.', correct: false, explain: '속도와 재현성은 별도의 acceptance gate입니다.' },
    ],
    shortPrompt: 'chunking, memory budget, streaming metric의 trade-off를 설명해 보세요.',
    expected: 'chunking은 peak memory를 낮추지만 metric 집계와 ordering/reproducibility를 보존해야 한다.',
  },
  '02_deep_learning/01_perceptron_and_mlp': {
    prompt: 'perceptron이 XOR에서 실패하는 핵심 이유는 무엇인가요?',
    explanation: '단일 선형 decision boundary는 XOR처럼 선형 분리되지 않는 패턴을 나눌 수 없습니다.',
    options: [
      { label: '하나의 직선/초평면 decision rule로는 XOR의 네 점을 올바르게 분리할 수 없다.', correct: true, explain: 'hidden layer와 nonlinearity가 필요한 이유입니다.' },
      { label: 'XOR은 입력 feature가 너무 많아서 perceptron이 실패한다.', correct: false, explain: '문제는 feature 수가 아니라 선형 분리 가능성입니다.' },
      { label: 'perceptron은 bias가 없을 때만 작동하고 bias가 있으면 항상 실패한다.', correct: false, explain: 'bias는 boundary 위치를 옮기지만 XOR의 비선형성은 해결하지 못합니다.' },
    ],
    shortPrompt: 'linear separability와 hidden layer의 관계를 설명해 보세요.',
    expected: '선형 분리 가능한 문제는 단일 boundary로 되지만 XOR은 hidden nonlinearity가 필요하다.',
  },
  '02_deep_learning/02_cnn_and_image_classification': {
    prompt: 'convolution을 local pattern detector라고 부를 수 있는 이유는 무엇인가요?',
    explanation: '작은 kernel이 이미지의 국소 영역을 훑으며 같은 weight로 반복 적용되어 feature map을 만듭니다.',
    options: [
      { label: 'kernel이 local receptive field에 반복 적용되어 위치별 feature map 반응을 만든다.', correct: true, explain: 'locality와 parameter sharing이 CNN의 inductive bias입니다.' },
      { label: 'convolution은 모든 픽셀을 별도 파라미터로 완전히 독립 처리한다.', correct: false, explain: '같은 kernel weight를 공유한다는 점이 중요합니다.' },
      { label: 'pooling은 channel 정보를 없애므로 classification에는 쓸 수 없다.', correct: false, explain: 'pooling은 공간 정보를 요약해 강건성을 줄 수 있습니다.' },
    ],
    shortPrompt: 'local receptive field와 parameter sharing을 이미지 분류 관점에서 설명해 보세요.',
    expected: '국소 패턴을 같은 kernel로 반복 감지해 위치 변화에 강한 feature map을 만든다.',
  },
  '02_deep_learning/03_sequence_models_rnn_lstm_gru': {
    prompt: '같은 token 집합도 순서가 바뀌면 final hidden state가 달라지는 이유는 무엇인가요?',
    explanation: 'RNN 계열은 이전 hidden state와 현재 token을 순차적으로 섞기 때문에 같은 원소라도 업데이트 순서가 결과를 바꿉니다.',
    options: [
      { label: 'hidden state가 시간 순서대로 갱신되어 이전 token 압축 상태가 다음 계산에 영향을 준다.', correct: true, explain: 'sequence ordering이 recurrent update의 핵심입니다.' },
      { label: 'RNN은 token 순서를 정렬해서 항상 같은 final state를 만든다.', correct: false, explain: '순서를 보존하는 것이 recurrent model의 중요한 특징입니다.' },
      { label: 'LSTM/GRU gate는 순서 정보를 완전히 제거하기 위해 존재한다.', correct: false, explain: 'gate는 정보를 선택적으로 보존/삭제하지만 순서성을 없애지는 않습니다.' },
    ],
    shortPrompt: 'hidden state 병목과 gating의 역할을 설명해 보세요.',
    expected: 'hidden state는 과거 정보를 압축하며, gate는 장기 정보 보존과 삭제를 조절한다.',
  },
  '02_deep_learning/04_attention_and_transformers': {
    prompt: 'attention output을 value들의 가중합이라고 말하는 직접 근거는 무엇인가요?',
    explanation: 'attention weight row가 softmax로 정규화되고 그 weight가 value matrix와 곱해져 sequence mixing 결과를 만듭니다.',
    options: [
      { label: '각 query row의 attention weight 합이 1이고 그 weight로 value를 섞어 output을 만든다.', correct: true, explain: 'row-stochastic weight와 value mixing이 핵심입니다.' },
      { label: 'attention은 value를 보지 않고 query와 key만 출력으로 사용한다.', correct: false, explain: 'query-key는 weight를 만들고 실제 내용은 value에서 옵니다.' },
      { label: 'causal mask는 padding token만 막고 미래 token은 항상 볼 수 있게 한다.', correct: false, explain: 'causal mask는 미래 위치를 보지 못하게 합니다.' },
    ],
    shortPrompt: 'Q/K/V와 mask가 attention output에 들어가는 순서를 설명해 보세요.',
    expected: 'QK score에 mask를 적용해 softmax weight를 만들고, 그 weight로 V를 섞는다.',
  },
  '02_deep_learning/05_autoencoders_and_representation_learning': {
    prompt: 'autoencoder의 bottleneck이 학습에 주는 압박은 무엇인가요?',
    explanation: 'encoder가 입력을 작은 latent로 압축하고 decoder가 복원해야 하므로 latent가 중요한 구조를 담아야 합니다.',
    options: [
      { label: 'latent 차원을 제한해 입력을 그대로 복사하지 못하게 하고 복원에 필요한 요약 표현을 배우게 한다.', correct: true, explain: 'bottleneck은 representation learning의 핵심 제약입니다.' },
      { label: 'decoder만 학습하면 encoder는 없어도 같은 latent를 얻는다.', correct: false, explain: 'encoder가 입력을 latent로 매핑해야 합니다.' },
      { label: 'reconstruction loss는 label이 없으면 계산할 수 없다.', correct: false, explain: '입력 자체를 target으로 삼을 수 있습니다.' },
    ],
    shortPrompt: 'encoder-latent-decoder와 reconstruction objective를 설명해 보세요.',
    expected: 'encoder가 입력을 latent로 압축하고 decoder가 복원하며 reconstruction loss가 학습 신호가 된다.',
  },
  '02_deep_learning/06_generative_models_vae_gan': {
    prompt: 'VAE에서 reparameterization trick이 필요한 이유는 무엇인가요?',
    explanation: '무작위 샘플링을 μ + σ·ε 형태로 분리해 latent sampling을 하면서도 μ/σ 경로로 gradient가 흐르게 합니다.',
    options: [
      { label: 'noise를 분리해 샘플링을 유지하면서 encoder가 만든 μ와 σ로 gradient가 전달되게 한다.', correct: true, explain: 'sampling과 backprop을 함께 가능하게 하는 장치입니다.' },
      { label: 'KL term을 제거해 reconstruction loss만 최적화하기 위해 필요하다.', correct: false, explain: 'VAE는 reconstruction과 KL 균형을 함께 봅니다.' },
      { label: 'GAN의 discriminator를 없애기 위해 쓰는 trick이다.', correct: false, explain: 'reparameterization은 VAE latent sampling의 문제입니다.' },
    ],
    shortPrompt: 'VAE의 reconstruction/KL 균형과 GAN의 mode coverage를 비교해 보세요.',
    expected: 'VAE는 복원과 latent prior 정렬을 함께 보며, GAN은 generator/discriminator 경쟁과 mode collapse 위험을 본다.',
  },
  '02_deep_learning/07_training_recipes_and_debugging': {
    prompt: '학습 레시피를 비교할 때 train loss만 보면 안 되는 이유는 무엇인가요?',
    explanation: 'overfit/underfit/divergence/data bug는 validation 곡선, sanity check, gradient/learning-rate 패턴을 같이 봐야 드러납니다.',
    options: [
      { label: 'train/validation 격차와 sanity check를 함께 봐야 overfit, divergence, label bug를 구분할 수 있다.', correct: true, explain: 'debugging은 단일 loss 감소보다 실패 유형 분류가 중요합니다.' },
      { label: 'train loss가 내려가면 validation과 data bug는 확인하지 않아도 된다.', correct: false, explain: 'train만 좋아지는 전형적 실패가 많습니다.' },
      { label: 'learning rate가 높을수록 항상 더 빨리 안정적으로 수렴한다.', correct: false, explain: '너무 큰 LR은 divergence를 만들 수 있습니다.' },
    ],
    shortPrompt: 'overfit/underfit/divergence/data bug를 어떤 증거로 나눌지 적어 보세요.',
    expected: 'train/val 곡선, sanity check, gradient/LR 흔적을 함께 보고 실패 유형을 분류한다.',
  },
  '03_nlp_bridge/01_tokenization_and_embeddings': {
    prompt: '공백 단어 수보다 subword token 수가 늘어나는 이유는 무엇인가요?',
    explanation: 'tokenizer는 단어를 vocabulary에 맞는 subword 조각으로 나누고, unknown/padding/mask 처리가 embedding 입력 길이를 바꿉니다.',
    options: [
      { label: '낯선 단어나 형태가 vocabulary subword 조각으로 분해되어 한 단어가 여러 token id가 될 수 있다.', correct: true, explain: 'subword tokenization의 핵심입니다.' },
      { label: 'embedding lookup이 단어를 문장 개수만큼 복사하기 때문이다.', correct: false, explain: '길이 증가는 lookup이 아니라 tokenization 단계에서 생깁니다.' },
      { label: 'padding mask는 token 수를 줄이기 위해 실제 token을 삭제한다.', correct: false, explain: 'padding mask는 padding 위치를 계산에서 제외하도록 표시합니다.' },
    ],
    shortPrompt: 'token id, embedding tensor, padding mask의 연결을 설명해 보세요.',
    expected: '문장은 token id sequence가 되고, embedding lookup 후 padding mask로 실제 token 위치만 해석한다.',
  },
  '03_nlp_bridge/02_attention_and_transformer_block': {
    prompt: 'padding mask와 causal mask는 각각 무엇을 막나요?',
    explanation: 'padding mask는 의미 없는 padding 위치를, causal mask는 현재 위치가 미래 token을 보는 것을 막습니다.',
    options: [
      { label: 'padding mask는 pad 위치 attention을 막고, causal mask는 미래 위치 attention을 막는다.', correct: true, explain: '두 mask의 목적이 다릅니다.' },
      { label: 'padding mask와 causal mask는 모두 embedding 차원을 줄이는 압축 연산이다.', correct: false, explain: 'mask는 attention score에 적용되는 접근 제한입니다.' },
      { label: 'causal mask는 encoder가 모든 token을 양방향으로 보게 하는 장치다.', correct: false, explain: 'causal mask는 decoder-style 미래 차단에 쓰입니다.' },
    ],
    shortPrompt: 'self-attention에서 Q/K/V와 mask가 shape를 어떻게 보존하는지 설명해 보세요.',
    expected: 'attention weight는 sequence 길이 축에서 섞고 output은 hidden dimension을 유지한다.',
  },
  '04_nlp/01_text_classification': {
    prompt: 'bag-of-words baseline이 긍정/부정 신호를 읽는 방식의 한계는 무엇인가요?',
    explanation: 'bag-of-words는 token 등장 신호를 보지만 순서, 문맥, 부정 표현의 조합을 놓치기 쉽습니다.',
    options: [
      { label: 'token 빈도 신호는 잡지만 순서와 문맥 조합을 잃어 특정 표현에서 오류가 날 수 있다.', correct: true, explain: 'baseline 해석과 neural classifier 비교 포인트입니다.' },
      { label: 'bag-of-words는 문장 순서를 완벽하게 보존하므로 문맥 오류가 없다.', correct: false, explain: '순서 정보를 버리는 것이 대표 한계입니다.' },
      { label: 'macro F1은 class별 성능 불균형을 숨기기 위해 쓰는 지표다.', correct: false, explain: 'macro F1은 오히려 class별 균형을 드러내는 데 유용합니다.' },
    ],
    shortPrompt: 'accuracy와 macro F1을 함께 봐야 하는 이유를 설명해 보세요.',
    expected: 'accuracy는 전체 정답률이고 macro F1은 class별 성능 균형을 더 잘 드러낸다.',
  },
  '04_nlp/02_named_entity_recognition': {
    prompt: 'BIO tagging에서 entity-level F1이 token accuracy와 다르게 중요한 이유는 무엇인가요?',
    explanation: '개체명은 시작/내부/경계가 맞아야 하나의 entity로 인정되므로 token 하나만 맞아도 충분하지 않습니다.',
    options: [
      { label: '경계와 label sequence가 맞아야 entity가 맞으므로 token별 정답률만으로는 boundary error를 숨길 수 있다.', correct: true, explain: 'NER의 핵심 오류는 경계와 alignment입니다.' },
      { label: 'BIO label은 모든 token에 같은 B label만 붙이면 된다.', correct: false, explain: 'B/I/O의 위치 규칙이 있습니다.' },
      { label: 'entity-level F1은 padding token까지 모두 정답으로 세는 지표다.', correct: false, explain: 'padding은 평가에서 제외되어야 합니다.' },
    ],
    shortPrompt: 'label alignment와 BIO boundary error를 설명해 보세요.',
    expected: 'subword/token 정렬 후 B/I/O 경계가 맞아야 entity-level 정답으로 인정된다.',
  },
  '04_nlp/03_machine_reading_comprehension': {
    prompt: 'exact match와 token F1을 함께 보면 어떤 span extraction 오류가 드러나나요?',
    explanation: '정답 span이 거의 맞아도 경계가 조금 틀리면 exact match는 실패하고 token F1은 부분 일치를 보여줍니다.',
    options: [
      { label: 'span 경계가 일부 틀린 partial match와 완전 오답을 구분할 수 있다.', correct: true, explain: '독해 평가는 boundary error를 따로 읽어야 합니다.' },
      { label: 'token F1이 있으면 no-answer threshold는 필요 없다.', correct: false, explain: 'answerable/unanswerable 판단은 별도 기준입니다.' },
      { label: 'question-context overlap은 항상 정답 span을 완벽히 보장한다.', correct: false, explain: 'heuristic overlap은 실패할 수 있습니다.' },
    ],
    shortPrompt: 'span extraction과 no-answer 판단을 함께 설명해 보세요.',
    expected: '모델은 answer span 경계와 답변 가능성/no-answer threshold를 함께 판단해야 한다.',
  },
  '05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives': {
    prompt: 'causal LM, masked LM, span corruption의 가장 큰 차이는 무엇인가요?',
    explanation: '각 objective는 입력으로 보이는 token과 loss를 계산하는 target token의 위치가 다릅니다.',
    options: [
      { label: '무엇을 가리고/남기고/다음 token으로 예측할지의 target framing과 scored token이 다르다.', correct: true, explain: 'pretraining objective의 핵심 구분입니다.' },
      { label: '세 objective는 모두 같은 token을 같은 위치에서 loss로 계산한다.', correct: false, explain: 'loss-mask density와 target 위치가 달라집니다.' },
      { label: 'span corruption은 sentinel token 없이 단어 순서를 무작위로 섞는 작업이다.', correct: false, explain: 'span을 sentinel로 대체하고 복원하는 framing입니다.' },
    ],
    shortPrompt: 'target framing과 loss-mask density를 연결해 설명해 보세요.',
    expected: 'objective마다 입력으로 남기는 context와 loss를 매기는 token 위치/밀도가 다르다.',
  },
  '05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture': {
    prompt: 'corpus quality가 단순히 데이터 규모와 같지 않은 이유는 무엇인가요?',
    explanation: '노이즈, 중복, contamination, domain imbalance, tokenizer coverage가 실제 학습 신호를 바꿉니다.',
    options: [
      { label: '중복/오염/도메인 불균형/토큰화 coverage가 token budget의 유효 학습 신호를 바꾼다.', correct: true, explain: 'data pipeline은 양보다 신호 품질이 중요합니다.' },
      { label: '문서 수가 많으면 contamination과 중복은 자동으로 희석되어 사라진다.', correct: false, explain: '오히려 반복 신호가 학습을 왜곡할 수 있습니다.' },
      { label: 'vocabulary size를 키우면 multilingual fairness 문제는 항상 해결된다.', correct: false, explain: 'compression과 coverage trade-off가 남습니다.' },
    ],
    shortPrompt: 'deduplication, contamination, domain balance를 token budget과 연결해 보세요.',
    expected: '같은 token budget에서도 품질/중복/오염/도메인 비율이 실제 학습 신호를 결정한다.',
  },
  '05_advanced_nlp_llm/03_domain_adaptive_pretraining': {
    prompt: 'DAPT가 일반 fine-tuning과 다른 핵심은 무엇인가요?',
    explanation: 'DAPT는 같은 pretraining objective를 유지한 채 domain corpus로 continued pretraining을 하며 specialization과 forgetting을 함께 봅니다.',
    options: [
      { label: '기존 LM objective를 유지하면서 domain corpus로 계속 사전학습해 domain gain과 retention을 같이 관리한다.', correct: true, explain: 'DAPT의 핵심 trade-off입니다.' },
      { label: 'DAPT는 classifier head만 바꾸는 supervised fine-tuning이다.', correct: false, explain: '같은 pretraining objective를 유지하는 것이 다릅니다.' },
      { label: 'domain loss가 낮아지면 general retention은 확인하지 않아도 된다.', correct: false, explain: 'catastrophic forgetting 위험을 봐야 합니다.' },
    ],
    shortPrompt: 'specialization gain과 catastrophic forgetting을 함께 설명해 보세요.',
    expected: 'domain 성능은 좋아질 수 있지만 일반 능력 유지/retention guardrail을 함께 확인해야 한다.',
  },
  '05_advanced_nlp_llm/04_instruction_tuning_and_sft': {
    prompt: 'SFT가 base LM의 continuation behavior를 assistant interaction으로 바꾸는 방식은 무엇인가요?',
    explanation: 'instruction/chat template으로 system/user/assistant role을 구성하고 assistant response token에 next-token loss를 적용합니다.',
    options: [
      { label: 'role/template로 입력-출력 형식을 만들고 assistant 답변 target을 next-token loss로 모방한다.', correct: true, explain: 'SFT는 objective는 유지하되 target framing을 바꿉니다.' },
      { label: 'SFT는 reward model 없이도 선호 비교를 직접 최적화하는 RL 알고리즘이다.', correct: false, explain: 'SFT는 supervised imitation 단계입니다.' },
      { label: 'chat template은 학습과 무관한 출력 꾸미기라 loss mask와 관련 없다.', correct: false, explain: '어떤 token을 target으로 삼을지에 직접 연결됩니다.' },
    ],
    shortPrompt: 'instruction format, chat template, assistant target을 연결해 보세요.',
    expected: 'prompt role framing을 만들고 assistant 응답 token을 supervised target으로 모방한다.',
  },
  '05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto': {
    prompt: 'chosen/rejected pair가 일반 정답/오답 label과 다른 점은 무엇인가요?',
    explanation: 'preference optimization은 한 답만 맞히는 것이 아니라 chosen 답의 log-prob를 rejected보다 더 높이도록 margin을 만듭니다.',
    options: [
      { label: '두 후보의 상대 선호를 사용해 chosen-rejected log-prob margin을 키우는 방향을 본다.', correct: true, explain: 'DPO/ORPO/KTO의 공통 감각입니다.' },
      { label: 'chosen은 정답 token이고 rejected는 항상 문법적으로 불가능한 token이다.', correct: false, explain: '둘 다 그럴듯한 응답일 수 있으며 선호 차이를 학습합니다.' },
      { label: 'preference optimization은 policy log-prob를 보지 않고 accuracy만 비교한다.', correct: false, explain: 'log-prob margin이 핵심입니다.' },
    ],
    shortPrompt: 'DPO/ORPO/KTO가 full RL loop 없이 margin을 움직이는 감각을 설명해 보세요.',
    expected: '선호 쌍 또는 desirability label로 policy가 선호 응답을 더 높은 확률로 두도록 조정한다.',
  },
  '05_advanced_nlp_llm/06_rlhf_and_reasoning_rl': {
    prompt: 'reward model을 truth engine이 아니라 preference proxy로 읽어야 하는 이유는 무엇인가요?',
    explanation: 'reward model은 인간/평가자 선호를 근사한 점수 신호이므로 policy update와 regression eval로 오용을 감시해야 합니다.',
    options: [
      { label: 'reward는 선호 근사 신호라 rollout, reward scoring, policy update, regression eval을 함께 관리해야 한다.', correct: true, explain: 'RLHF loop의 안전한 해석입니다.' },
      { label: 'reward가 높으면 답이 항상 사실이며 별도 평가가 필요 없다.', correct: false, explain: 'reward hacking과 preference proxy 오류가 가능합니다.' },
      { label: 'PPO/RLHF는 rollout 없이 정답 label만으로 끝나는 supervised loop다.', correct: false, explain: 'rollout과 reward scoring이 들어갑니다.' },
    ],
    shortPrompt: 'rollout→reward scoring→policy update→regression eval 순서를 설명해 보세요.',
    expected: '정책이 응답을 만들고 reward/judge가 점수화하며, 업데이트 후 회귀/안전 평가로 drift를 확인한다.',
  },
  '05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval': {
    prompt: 'RAG에서 citation 개수보다 claim-level evidence support가 중요한 이유는 무엇인가요?',
    explanation: '인용이 많아도 각 주장과 실제 근거가 맞지 않으면 grounded answer라고 보기 어렵습니다.',
    options: [
      { label: '각 claim이 검색된 근거로 실제 지지되는지 봐야 citation-without-support 오류를 잡을 수 있다.', correct: true, explain: 'grounding 평가는 claim 단위 근거 대응이 핵심입니다.' },
      { label: 'citation 태그가 있으면 문장 내용은 검색 결과와 무관해도 된다.', correct: false, explain: '형식적 인용만으로는 groundedness를 보장하지 않습니다.' },
      { label: 'retriever recall이 낮아도 generator가 항상 사실을 복구한다.', correct: false, explain: '외부 memory 실패는 hallucination 위험을 키웁니다.' },
    ],
    shortPrompt: 'retriever-reader와 retriever-generator의 failure pattern 차이를 설명해 보세요.',
    expected: 'reader는 근거에서 답을 추출하고 generator는 근거를 문맥으로 생성하므로 retrieval/grounding 실패 양상이 다르다.',
  },
  '05_advanced_nlp_llm/08_alignment_safety_and_model_behavior': {
    prompt: 'alignment와 capability를 분리하지 않으면 어떤 평가 착시가 생기나요?',
    explanation: '능력 점수가 높아도 unsafe compliance나 over-refusal이 숨을 수 있으므로 behavior slice를 나눠 봐야 합니다.',
    options: [
      { label: 'capability score가 높은 모델도 harmful compliance나 over-refusal을 보일 수 있어 slice별 행동 평가가 필요하다.', correct: true, explain: 'alignment-vs-capability 분리의 이유입니다.' },
      { label: '정답률이 높으면 safety behavior도 자동으로 안전하다고 볼 수 있다.', correct: false, explain: '능력과 안전 행동은 별도 축입니다.' },
      { label: 'refusal은 많을수록 항상 좋고 benign 요청 거절은 문제가 아니다.', correct: false, explain: 'over-refusal은 사용성/정렬 실패입니다.' },
    ],
    shortPrompt: 'refusal, over-refusal, unsafe compliance를 구분해 보세요.',
    expected: 'harmful 요청 거절은 필요하지만 benign 요청 거절은 over-refusal이고 harmful 수락은 unsafe compliance다.',
  },
  '06_training_systems/01_torchrun_and_ddp_basics': {
    prompt: 'rank와 local rank를 구분해야 하는 이유는 무엇인가요?',
    explanation: 'global rank는 전체 process identity이고 local rank는 한 노드 안의 device 배치에 연결됩니다. DDP는 replica gradient를 all-reduce로 평균냅니다.',
    options: [
      { label: 'global rank는 전체 worker 식별, local rank는 노드 내부 device 배치라 실행/로그/통신 해석이 달라진다.', correct: true, explain: 'torchrun 실행 계약의 기본입니다.' },
      { label: 'rank와 local rank는 항상 같은 값이므로 구분할 필요가 없다.', correct: false, explain: 'multi-node에서 달라질 수 있습니다.' },
      { label: 'DDP는 모델을 shard하고 parameter를 rank마다 나눠 저장한다.', correct: false, explain: '기본 DDP는 replica를 두고 gradient를 평균냅니다.' },
    ],
    shortPrompt: 'DDP가 무엇을 복제하고 무엇을 평균내는지 설명해 보세요.',
    expected: '모델 replica는 rank마다 있고 batch를 나눠 계산한 gradient를 all-reduce로 평균낸다.',
  },
  '06_training_systems/02_accelerate_workflows': {
    prompt: 'Accelerate가 줄여 주는 것과 여전히 사용자가 알아야 하는 것은 무엇인가요?',
    explanation: 'Accelerator는 device placement, prepare, mixed precision, distributed launch boilerplate를 줄이지만 batch/loss/metric 의미는 사용자가 이해해야 합니다.',
    options: [
      { label: '장치 배치와 분산 준비 코드를 줄여도 batch, loss scaling, metric 집계 의미는 직접 이해해야 한다.', correct: true, explain: 'prepare 이후에도 훈련 계약은 사라지지 않습니다.' },
      { label: 'Accelerate를 쓰면 optimizer/loss/metric 정의를 몰라도 자동으로 올바른 실험이 된다.', correct: false, explain: '도구는 boilerplate를 줄일 뿐 의미 해석을 대신하지 않습니다.' },
      { label: 'mixed precision은 정확도와 overflow 문제를 완전히 제거한다.', correct: false, explain: '스케일링과 안정성 관찰이 필요합니다.' },
    ],
    shortPrompt: '`prepare()` 이후에도 남는 사용자 책임을 설명해 보세요.',
    expected: '모델/optimizer/dataloader는 감싸지지만 batch 의미, loss 정규화, metric 집계는 이해해야 한다.',
  },
  '06_training_systems/03_deepspeed_zero': {
    prompt: 'ZeRO stage를 memory accounting으로 읽을 때 핵심은 무엇인가요?',
    explanation: 'stage가 올라갈수록 optimizer state, gradient, parameter 중 어떤 상태를 data parallel rank 사이에 shard하는지가 달라집니다.',
    options: [
      { label: 'stage별로 optimizer state→gradient→parameter sharding 범위가 넓어져 per-rank memory와 communication trade-off가 바뀐다.', correct: true, explain: 'ZeRO의 핵심은 중복 상태 제거입니다.' },
      { label: 'ZeRO는 batch를 작게 만들어 activation을 전부 제거하는 기법이다.', correct: false, explain: '주로 data parallel 중복 상태를 shard합니다.' },
      { label: 'stage가 높을수록 통신 비용은 항상 0이 된다.', correct: false, explain: '메모리를 줄이는 대신 gather/scatter 통신이 늘 수 있습니다.' },
    ],
    shortPrompt: 'ZeRO가 shard하는 상태와 trade-off를 설명해 보세요.',
    expected: 'optimizer/gradient/parameter 상태 중복을 줄여 per-rank memory를 낮추지만 통신/복잡도 비용이 생긴다.',
  },
  '06_training_systems/04_fsdp_checkpointing_and_offload': {
    prompt: 'FSDP runtime을 parameter shard lifecycle로 읽는다는 뜻은 무엇인가요?',
    explanation: 'FSDP는 필요한 순간 parameter를 all-gather하고 계산 후 다시 shard하며, checkpoint/offload/checkpointing이 memory-compute-I/O trade-off를 만듭니다.',
    options: [
      { label: 'forward/backward 계산 때 parameter를 모으고 이후 다시 shard하며 checkpoint/offload 정책이 memory와 I/O를 바꾼다.', correct: true, explain: 'FSDP의 runtime 감각입니다.' },
      { label: 'FSDP는 parameter를 한 번 모은 뒤 학습 내내 모든 rank에 full copy로 유지한다.', correct: false, explain: 'shard lifecycle을 놓친 설명입니다.' },
      { label: 'activation checkpointing은 checkpoint 파일 저장만 빠르게 하는 기능이다.', correct: false, explain: 'activation 재계산으로 메모리를 줄이는 기법입니다.' },
    ],
    shortPrompt: 'activation checkpointing과 CPU offload의 비용을 설명해 보세요.',
    expected: 'checkpointing은 activation 저장을 줄이고 recomputation을 늘리며, offload는 GPU 메모리를 줄이는 대신 I/O/전송 비용을 만든다.',
  },
  '06_training_systems/05_tensor_parallelism': {
    prompt: 'tensor parallelism이 state sharding과 다른 점은 무엇인가요?',
    explanation: 'tensor parallelism은 레이어 내부 행렬/attention head 계산 자체를 나눠 collective로 합칩니다.',
    options: [
      { label: 'parameter 상태만 저장 분할하는 것이 아니라 row/column linear나 attention head 같은 intra-layer 계산을 나눈다.', correct: true, explain: 'intra-layer parallelism 감각이 핵심입니다.' },
      { label: 'tensor parallelism은 batch sample을 rank마다 나누는 data parallel과 같다.', correct: false, explain: 'batch 축이 아니라 모델 내부 tensor 축을 나눕니다.' },
      { label: 'row/column parallel은 dense 결과와 수치 검산이 필요 없다.', correct: false, explain: 'shard 합산 결과가 dense와 맞는지 확인해야 합니다.' },
    ],
    shortPrompt: 'row parallel과 column parallel linear의 검산 기준을 설명해 보세요.',
    expected: '나눈 weight/activation shard를 collective로 모았을 때 dense matmul과 max diff가 작아야 한다.',
  },
  '06_training_systems/06_pipeline_parallelism': {
    prompt: 'pipeline parallelism이 data/tensor parallel과 다른 축은 무엇인가요?',
    explanation: 'pipeline parallelism은 레이어 stack을 stage로 나누고 microbatch schedule로 시간축 실행을 채웁니다.',
    options: [
      { label: '모델 레이어를 stage로 나누고 microbatch schedule로 bubble/throughput trade-off를 관리한다.', correct: true, explain: 'partition과 schedule이 핵심입니다.' },
      { label: 'pipeline은 batch sample만 나누며 모델 레이어는 모든 rank에 완전히 복제한다.', correct: false, explain: '레이어 stage partition이 pipeline의 특징입니다.' },
      { label: 'stage를 나누면 single-batch latency가 항상 줄어든다.', correct: false, explain: 'bubble과 stage imbalance 때문에 latency/throughput을 따로 봐야 합니다.' },
    ],
    shortPrompt: 'microbatch, bubble, stage balance를 연결해 설명해 보세요.',
    expected: 'microbatch로 stage를 채워 throughput을 높이지만 bubble과 imbalance가 효율을 제한한다.',
  },
  '06_training_systems/07_data_parallel_grad_accumulation': {
    prompt: 'data parallel과 gradient accumulation은 batch를 어떻게 다르게 키우나요?',
    explanation: 'data parallel은 여러 rank의 local batch를 합쳐 global batch를 만들고, grad accumulation은 optimizer step 전에 여러 microbatch gradient를 누적합니다.',
    options: [
      { label: 'data parallel은 rank 축으로 global batch를 키우고 accumulation은 optimizer step cadence를 늦춰 effective batch를 키운다.', correct: true, explain: 'local/global/effective batch 구분이 핵심입니다.' },
      { label: 'grad accumulation은 매 microbatch마다 optimizer step을 더 자주 하게 만든다.', correct: false, explain: '오히려 step을 지연합니다.' },
      { label: 'no_sync는 gradient 계산 자체를 끄는 기능이다.', correct: false, explain: '동기화를 지연할 뿐 gradient 계산은 계속됩니다.' },
    ],
    shortPrompt: 'local batch, global batch, effective batch를 구분해 보세요.',
    expected: 'local은 rank당 batch, global은 rank 합산 batch, effective는 accumulation step까지 곱한 업데이트 기준 batch다.',
  },
  '06_training_systems/08_hybrid_parallel_topologies': {
    prompt: 'hybrid parallel topology가 단순 옵션 조합이 아닌 이유는 무엇인가요?',
    explanation: 'DP/TP/PP/FSDP는 각각 나누는 축과 통신 병목이 다르고 hardware link와 checkpoint/recovery 계약까지 함께 결정합니다.',
    options: [
      { label: '모델 크기, 메모리 병목, 통신 경로, 하드웨어 링크, checkpoint 계약을 같이 맞추는 배치 설계다.', correct: true, explain: 'hybrid topology는 설계 문제입니다.' },
      { label: '모든 parallelism 옵션을 켜면 항상 최적 topology가 된다.', correct: false, explain: '서로 다른 통신/메모리 비용이 충돌할 수 있습니다.' },
      { label: 'TP와 PP를 쓰면 DP/FSDP checkpoint metadata는 필요 없다.', correct: false, explain: '복구와 재배치를 위해 topology-aware metadata가 필요합니다.' },
    ],
    shortPrompt: 'DP, TP, PP, FSDP가 각각 무엇을 나누는지 설명해 보세요.',
    expected: 'DP는 batch/replica, TP는 layer tensor, PP는 layer stage, FSDP는 state shard를 나눈다.',
  },
  '06_training_systems/09_profiling_monitoring_and_failure_recovery': {
    prompt: '느린 학습 run을 time/memory/communication 축으로 나누면 왜 triage가 빨라지나요?',
    explanation: '병목 가설을 step timeline, memory snapshot, communication wait, heartbeat/failure signal로 분리해 볼 수 있습니다.',
    options: [
      { label: 'step timeline과 memory/communication 신호를 분리하면 OOM, hang, divergence, slow phase를 다른 가설로 좁힐 수 있다.', correct: true, explain: 'runbook식 triage의 목적입니다.' },
      { label: 'average step time 하나만 보면 jitter와 tail latency까지 모두 설명된다.', correct: false, explain: '평균은 phase-boundary slowdown과 tail을 숨깁니다.' },
      { label: 'checkpoint resume 문제는 profiling/monitoring과 무관하다.', correct: false, explain: 'failure recovery의 핵심 운영 신호입니다.' },
    ],
    shortPrompt: 'OOM, hang, divergence, checkpoint resume을 어떤 증거로 나눌지 적어 보세요.',
    expected: '메모리 snapshot, heartbeat/timeout, loss/gradient 추세, checkpoint metadata와 resume log로 나눠 본다.',
  },
  '07_frontier_labs/01_paper_reproduction_playground': {
    prompt: 'full paper reproduction과 claim-level reproduction은 어떻게 다른가요?',
    explanation: 'claim-level reproduction은 논문 전체 복제가 아니라 특정 claim을 evidence matrix와 제한된 scope로 검증합니다.',
    options: [
      { label: '핵심 claim을 작게 자르고 baseline/reported/reproduced evidence를 나란히 비교한다.', correct: true, explain: '제약 있는 재현에서 정직한 범위 설정입니다.' },
      { label: '논문 전체를 완전히 복제하지 못하면 어떤 claim도 검증할 수 없다.', correct: false, explain: 'claim-level scope가 바로 이를 해결합니다.' },
      { label: 'reported result와 reproduced result가 다르면 항상 코드가 틀렸다는 뜻이다.', correct: false, explain: 'variance, 환경, 데이터 차이 가설을 함께 봐야 합니다.' },
    ],
    shortPrompt: 'claim/evidence matrix와 scope control을 설명해 보세요.',
    expected: '검증할 claim, 필요한 evidence, baseline/reported/reproduced 비교, mismatch 가설을 명시한다.',
  },
  '07_frontier_labs/02_capstone_model_building': {
    prompt: 'capstone 아이디어를 어디까지 줄여야 “끝낼 수 있는 scope”가 되나요?',
    explanation: 'problem statement, non-goal, dataset/model/eval contract, milestone, risk register가 명시될 때 실행 가능한 프로젝트가 됩니다.',
    options: [
      { label: '문제, 하지 않을 것, 데이터/모델/평가 계약, acceptance gate와 risk를 한 장으로 고정할 수 있을 만큼 줄인다.', correct: true, explain: 'capstone은 화려함보다 끝낼 수 있는 계약이 중요합니다.' },
      { label: '가장 큰 모델과 가장 넓은 데이터셋을 쓰면 scope는 자동으로 명확해진다.', correct: false, explain: '범위가 커질수록 실패/해석 위험이 커집니다.' },
      { label: 'baseline은 멋진 최종 모델이 준비된 뒤에만 정한다.', correct: false, explain: 'baseline은 처음부터 비교선을 제공합니다.' },
    ],
    shortPrompt: 'problem statement, non-goal, eval contract를 한 문장으로 연결해 보세요.',
    expected: '무엇을 풀지/풀지 않을지와 성공 판정 기준을 먼저 고정해야 한다.',
  },
  '07_frontier_labs/03_agentic_training_and_eval_loops': {
    prompt: 'agentic training/eval loop가 단순 job automation과 다른 점은 무엇인가요?',
    explanation: 'planner/executor/verifier/critic 역할을 나누고 evidence-first stop/escalation 기준을 두어 self-approval을 막습니다.',
    options: [
      { label: '실험 계약 아래 역할별 판단과 검증 기준을 분리해 반복의 증거와 중단 조건을 남긴다.', correct: true, explain: 'agentic loop는 자동 반복보다 검증 구조가 핵심입니다.' },
      { label: 'agent가 있으면 verifier 없이도 모든 반복을 자동 승인해도 된다.', correct: false, explain: 'self-approval 위험이 큽니다.' },
      { label: 'retry budget과 escalation 기준은 성공률을 낮추므로 제거해야 한다.', correct: false, explain: '무한 반복과 drift를 막는 안전장치입니다.' },
    ],
    shortPrompt: 'planner/executor/verifier/critic 분리의 이유를 설명해 보세요.',
    expected: '계획, 실행, 검증, 비판을 분리해 근거 없는 자기 승인과 반복 drift를 줄인다.',
  },
  '07_frontier_labs/04_benchmark_and_dataset_construction': {
    prompt: 'task contract를 먼저 고정하면 benchmark claim이 왜 선명해지나요?',
    explanation: 'unit of record, schema, split, annotation rubric, leakage/contamination audit가 score가 말할 수 있는 범위를 정합니다.',
    options: [
      { label: '데이터 단위, split, annotation 기준, leakage audit를 고정해 score가 주장할 수 있는 boundary를 제한한다.', correct: true, explain: 'benchmark card가 필요한 이유입니다.' },
      { label: 'score가 높으면 task definition과 dataset schema는 나중에 써도 된다.', correct: false, explain: '정의 없는 score는 해석할 수 없습니다.' },
      { label: 'holdout은 데이터가 적을 때 생략해야 leakage 위험이 줄어든다.', correct: false, explain: 'holdout/split hygiene가 leakage 방어의 핵심입니다.' },
    ],
    shortPrompt: 'benchmark card에 들어갈 source/split/schema/QC 항목을 설명해 보세요.',
    expected: 'source boundary, unit of record, split manifest, annotation rubric, leakage/contamination audit를 남긴다.',
  },
  '07_frontier_labs/05_open_ended_research_tracks': {
    prompt: 'open-ended research를 자유 탐색이 아니라 운영 문제로 보는 이유는 무엇인가요?',
    explanation: 'north-star question, hypothesis registry, iteration boundary, kill criteria, evidence standard가 없으면 탐색이 끝나지 않습니다.',
    options: [
      { label: '작은 연구 범위와 keep/kill 기준을 정해야 반복을 증거 기반으로 멈추거나 이어갈 수 있다.', correct: true, explain: 'open-ended일수록 운영 계약이 필요합니다.' },
      { label: '열린 연구에서는 kill criteria를 두면 창의성이 사라져 쓰면 안 된다.', correct: false, explain: '기준 없이는 끝나지 않는 탐색이 됩니다.' },
      { label: 'hypothesis registry는 성공한 실험만 기록하는 홍보 문서다.', correct: false, explain: '실패와 반증 조건까지 기록해야 합니다.' },
    ],
    shortPrompt: 'hypothesis registry와 kill criteria를 연결해 설명해 보세요.',
    expected: '가설, mechanism, evidence standard, iteration boundary, kill/keep 판단 기준을 함께 기록한다.',
  },
  '08_multimodal_bridge/01_contrastive_alignment': {
    prompt: 'contrastive alignment에서 정답 이미지-텍스트 쌍이 similarity matrix 대각선에 놓인다는 뜻은 무엇인가요?',
    explanation: '같은 index의 image/text embedding이 positive pair이고, 나머지는 negative로 비교되어 retrieval ranking을 만듭니다.',
    options: [
      { label: '같은 index의 image-text positive similarity가 negative보다 높아져 양방향 retrieval 순위가 좋아져야 한다.', correct: true, explain: 'joint embedding alignment의 핵심입니다.' },
      { label: '대각선 값은 항상 낮아야 hard negative를 잘 구분한다.', correct: false, explain: 'positive pair는 보통 높아져야 합니다.' },
      { label: 'temperature는 similarity matrix와 무관한 시각화 색상 설정이다.', correct: false, explain: 'temperature는 softmax/logit scale에 영향을 줍니다.' },
    ],
    shortPrompt: 'positive/negative similarity와 Recall@K를 연결해 설명해 보세요.',
    expected: 'positive pair가 negative보다 높게 rank되어야 image→text/text→image retrieval 성능이 오른다.',
  },
  '09_multimodal/01_image_text_retrieval': {
    prompt: 'Recall@1과 Recall@2를 함께 읽으면 어떤 ranking 정보를 얻나요?',
    explanation: '정답이 1등인지, 아니면 가까운 후보 안에는 있지만 top-1에서는 밀렸는지 구분할 수 있습니다.',
    options: [
      { label: '정답이 top-1에 있는지와 top-k 안에는 들어오는지 구분해 hard negative로 인한 순위 밀림을 볼 수 있다.', correct: true, explain: 'retrieval은 순위 지표를 함께 봐야 합니다.' },
      { label: 'Recall@2가 높으면 Recall@1은 항상 같은 값이다.', correct: false, explain: 'top-k가 커지면 더 쉬운 기준이 됩니다.' },
      { label: 'image→text와 text→image 난이도는 항상 완전히 같다.', correct: false, explain: 'query/candidate 구조에 따라 비대칭일 수 있습니다.' },
    ],
    shortPrompt: 'hard negative와 bidirectional retrieval 실패를 설명해 보세요.',
    expected: '유사한 오답이 정답보다 위에 오를 수 있고, image→text/text→image 방향별 ranking 난이도가 다를 수 있다.',
  },
  '09_multimodal/02_image_captioning': {
    prompt: 'captioning에서 자동 지표가 괜찮아도 hallucination 사례를 봐야 하는 이유는 무엇인가요?',
    explanation: 'BLEU 같은 표면 지표가 일부 n-gram을 맞춰도 이미지에 없는 객체/속성을 생성할 수 있습니다.',
    options: [
      { label: '표면 token overlap이 높아도 이미지 근거 없는 객체나 속성을 생성하는지 사람이 사례를 확인해야 한다.', correct: true, explain: 'captioning 평가는 metric과 qualitative failure를 함께 봅니다.' },
      { label: 'BLEU-1이 높으면 이미지 grounding은 자동으로 보장된다.', correct: false, explain: '토큰 overlap과 시각 근거는 다릅니다.' },
      { label: 'teacher forcing으로 학습하면 greedy decoding 오류는 생기지 않는다.', correct: false, explain: '학습/평가 decoding gap이 남습니다.' },
    ],
    shortPrompt: 'teacher forcing과 greedy decoding gap을 설명해 보세요.',
    expected: '학습 때는 정답 prefix를 보지만 평가 때는 자기 예측을 이어가므로 오류가 누적될 수 있다.',
  },
  '09_multimodal/03_visual_question_answering': {
    prompt: 'VQA에서 overall accuracy만 보면 놓치는 것은 무엇인가요?',
    explanation: 'answer type별로 count/color/yes-no 문제가 다르게 실패할 수 있고 shortcut bias가 특정 유형을 숨길 수 있습니다.',
    options: [
      { label: 'answer type breakdown을 봐야 count, color, yes/no별 grounded reasoning failure와 shortcut bias를 구분한다.', correct: true, explain: 'VQA는 유형별 실패 분석이 중요합니다.' },
      { label: 'overall accuracy가 높으면 모든 answer type이 동일하게 잘 된 것이다.', correct: false, explain: '유형별 실패가 평균에 숨을 수 있습니다.' },
      { label: '질문 token만 보면 이미지 grounding은 필요 없다.', correct: false, explain: 'VQA는 이미지와 질문을 함께 읽어야 합니다.' },
    ],
    shortPrompt: 'shortcut bias와 grounded reasoning failure를 구분해 보세요.',
    expected: '언어/데이터 편향만으로 맞히는 shortcut과 이미지 근거를 실제로 확인하지 못하는 실패를 나눠 본다.',
  },
  '10_vla/01_vision_language_action_grounding': {
    prompt: 'VQA answer와 VLA action token의 핵심 차이는 무엇인가요?',
    explanation: 'VQA는 질문에 대한 답을 생성/분류하지만 VLA는 시각 상태와 언어 지시를 실제 action 및 safety gate로 바꿉니다.',
    options: [
      { label: 'VLA action token은 환경에 영향을 주는 행동 선택이라 safety gate와 trajectory 성공률을 별도 지표로 봐야 한다.', correct: true, explain: '행동은 답변보다 더 강한 안전/성공 계약이 필요합니다.' },
      { label: 'VLA action token은 VQA의 텍스트 답변과 같아서 safety metric이 필요 없다.', correct: false, explain: '행동은 실행 결과와 안전 위험을 동반합니다.' },
      { label: 'behavior cloning은 trajectory 없이 단일 정답 단어만 외우는 작업이다.', correct: false, explain: '상태-지시-action trajectory mapping을 학습합니다.' },
    ],
    shortPrompt: 'action accuracy와 safety gate accuracy를 분리해야 하는 이유를 설명해 보세요.',
    expected: '맞는 action을 고르는 것과 위험한 action을 막는 것은 다른 실패를 잡는 별도 지표다.',
  },
};

function quizBlueprintForUnit(unit) {
  return LESSON_QUIZ_BLUEPRINTS[unit.path] || fallbackQuizBlueprint(unit);
}

function fallbackQuizBlueprint(unit) {
  const keyTerms = unit.key_terms || [];
  const analysis = unit.analysis_questions || [];
  const primaryTerm = keyTerms[0] || '핵심 개념';
  const secondaryTerm = keyTerms[1] || '실행 결과';
  const firstQuestion = analysis[0] || `${primaryTerm}이 실행 결과와 어떻게 연결되는가?`;
  return {
    prompt: `${unit.title}에서 “${firstQuestion}”에 답할 때 가장 먼저 피해야 할 해석은 무엇인가요?`,
    explanation: '단원별 전용 문항이 없을 때도 파일명 선택이 아니라 개념과 실행 근거를 연결하도록 묻습니다.',
    options: [
      { label: `${primaryTerm}와 ${secondaryTerm}를 코드의 지표/그림/해석 노트와 연결해 판단한다.`, correct: true, explain: '개념과 실행 근거를 같이 봐야 합니다.' },
      { label: '터미널에 글자가 출력됐으면 산출물과 지표는 확인하지 않는다.', correct: false, explain: '재확인 가능한 근거가 남지 않습니다.' },
      { label: '단원 제목만 외우고 코드의 입력/출력 흐름은 보지 않는다.', correct: false, explain: '코드 흐름 없이 개념이 실행 결과와 연결되지 않습니다.' },
    ],
    shortPrompt: `분석 질문에 자기 말로 답해 보세요: ${firstQuestion}`,
    expected: `${primaryTerm} 또는 ${secondaryTerm}를 사용해 실행 산출물에서 본 근거와 연결해 설명`,
  };
}

function quizQuestionFromBlueprint(blueprint) {
  return {
    id: 'concept-check',
    type: blueprint.type || 'single',
    prompt: blueprint.prompt,
    explanation: blueprint.explanation,
    options: (blueprint.options || []).map((option, index) => ({
      id: option.id || `option-${index}`,
      label: option.label,
      correct: Boolean(option.correct),
      explain: option.explain || '',
    })),
  };
}

function quizForUnit(unit) {
  const blueprint = quizBlueprintForUnit(unit);
  return [
    quizQuestionFromBlueprint(blueprint),
    {
      id: 'concept',
      type: 'short',
      prompt: blueprint.shortPrompt || `분석 질문에 자기 말로 답해 보세요: ${(unit.analysis_questions || [])[0] || unit.objective || unit.title}`,
      expected: blueprint.expected || '핵심 개념을 실행 결과, 지표, 그림, 해석 노트 중 하나와 연결해 설명',
      explanation: '짧은 답변은 자동 정답 하나로 고정하지 않습니다. 자기 말 설명을 남기고, 아래 기준과 비교하세요.',
      options: [],
    },
  ];
}

function renderQuizPanel(unit, quizItems, quizAnswers, wrongNotes) {
  const answered = quizItems.filter((item) => quizAnswers[item.id]).length;
  return `<section class="quiz-panel" aria-label="단원 점검 퀴즈">
    <h3>단원 점검 퀴즈 <span>${answered}/${quizItems.length} 완료</span></h3>
    ${quizItems.map((question) => renderQuizQuestion(question, quizAnswers[question.id], wrongNotes[question.id])).join('')}
  </section>`;
}

function renderQuizQuestion(question, answerState, wrongNote) {
  const stateClass = answerState ? (question.type === 'short' ? 'in_progress' : (answerState.correct ? 'done' : 'blocked')) : 'not_started';
  const submitLabel = question.type === 'short' ? '예시와 비교 저장' : '정답 확인';
  return `<article class="quiz-question ${stateClass}">
    <strong>${escapeHtml(question.prompt)}</strong>
    ${renderQuizInputs(question, answerState?.answer)}
    <button type="button" data-quiz-submit="${escapeHtml(question.id)}">${submitLabel}</button>
    ${answerState ? renderQuizFeedback(question, answerState, wrongNote) : ''}
  </article>`;
}

function renderQuizInputs(question, answer) {
  if (question.type === 'short') {
    return `<textarea data-quiz-id="${escapeHtml(question.id)}" rows="2" placeholder="짧게 자기 말로 적어 보세요.">${escapeHtml(answer || '')}</textarea>`;
  }
  const current = Array.isArray(answer) ? answer : [answer].filter(Boolean);
  const type = question.type === 'multi' ? 'checkbox' : 'radio';
  return `<div class="quiz-options">${question.options.map((option) => `<label><input type="${type}" name="quiz-${escapeHtml(question.id)}" data-quiz-id="${escapeHtml(question.id)}" value="${escapeHtml(option.id)}" ${current.includes(option.id) ? 'checked' : ''}/> ${escapeHtml(option.label)}</label>`).join('')}</div>`;
}

function renderQuizFeedback(question, answerState, wrongNote) {
  if (question.type === 'short') {
    return `<div class="quiz-feedback review">
      <strong>자동 채점 대신 예시와 비교하세요</strong>
      <p>${escapeHtml(question.explanation)}</p>
      <p><b>비교 기준:</b> ${escapeHtml(question.expected || question.explanation)}</p>
      <p>이 답변은 진행 기록에 저장되지만 “맞았습니다”로 처리하지 않습니다. 기준과 다르면 아래 내 메모나 오답노트에 복습 질문을 남기세요.</p>
    </div>`;
  }
  const correct = answerState.correct;
  const optionHints = (question.options || []).map((option) => `<li><strong>${escapeHtml(option.label)}</strong>: ${escapeHtml(option.explain || '')}</li>`).join('');
  return `<div class="quiz-feedback ${correct ? 'done' : 'blocked'}">
    <strong>${correct ? '맞았습니다' : '다시 확인'}</strong>
    <p>${escapeHtml(question.explanation)}</p>
    ${optionHints ? `<ul>${optionHints}</ul>` : `<p>비교 기준: ${escapeHtml(question.expected || question.explanation)}</p>`}
    ${!correct ? `<label>왜 헷갈렸나요?<textarea data-wrong-note-memo="${escapeHtml(question.id)}" rows="2" placeholder="오답 이유를 적으면 오답노트에 저장됩니다.">${escapeHtml(wrongNote?.memo || '')}</textarea></label>` : ''}
  </div>`;
}

function renderWrongNotesPanel(unit, wrongNotes) {
  const notes = Object.values(wrongNotes || {}).filter((note) => !note.recovered);
  if (!notes.length) return '<section class="wrong-note-panel"><h3>오답노트</h3><p class="empty">아직 열린 오답이 없습니다.</p></section>';
  return `<section class="wrong-note-panel" aria-label="단원 오답노트">
    <h3>오답노트</h3>
    ${notes.map((note) => `<article><strong>${escapeHtml(note.question)}</strong><p>내 답: ${escapeHtml(note.learnerAnswer)}</p><p>정답: ${escapeHtml(note.correctAnswer)}</p><p>${escapeHtml(note.memo || '오답 이유 메모를 아직 남기지 않았습니다.')}</p></article>`).join('')}
  </section>`;
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
      hint: unit.objective || '단원 안내 첫 단락을 자기 말로 바꿔 보세요.',
    },
    {
      id: 'run-observe',
      label: '코드 실행 결과에서 봐야 할 숫자와 산출물을 짚을 수 있다',
      hint: (unit.required_outputs || []).slice(0, 2).join(', ') || '지표 파일, 그림, 해석 노트 중 무엇이 남는지 확인하세요.',
    },
  ];
  (unit.analysis_questions || []).slice(0, 2).forEach((question, index) => {
    checks.push({
      id: `analysis-${index + 1}`,
      label: `분석 질문에 답할 수 있다: ${question}`,
      hint: '실행 관찰 카드와 해석 노트를 보고 2~3문장으로 답해 보세요.',
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
  if (labels.includes('run_stage.py')) return 'ML 단계 실습';
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
  $('#overall-progress').textContent = stats.percent ? `${stats.percent}% 완료` : '학습 전';
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

function allWrongNotes() {
  const lessons = currentUser().lessons || {};
  return Object.values(lessons)
    .flatMap((lesson) => Object.values(lesson.wrongNotes || {}))
    .filter((note) => note && !note.recovered)
    .sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
}

function openMistakeReview() {
  const notes = allWrongNotes();
  const container = $('#mistake-review');
  container.innerHTML = notes.length ? notes.map((note) => `<article class="mistake-card">
      <strong>${escapeHtml(note.unitTitle || note.unitPath)}</strong>
      <p>${escapeHtml(note.question)}</p>
      <p><b>내 답</b>: ${escapeHtml(note.learnerAnswer)}</p>
      <p><b>정답</b>: ${escapeHtml(note.correctAnswer)}</p>
      <p><b>메모</b>: ${escapeHtml(note.memo || '아직 메모 없음')}</p>
      <button type="button" data-mistake-unit="${escapeHtml(note.unitPath)}">이 단원 열기</button>
    </article>`).join('') : '<p class="empty">현재 열린 오답이 없습니다. 틀린 퀴즈가 생기면 여기에 모입니다.</p>';
  container.querySelectorAll('[data-mistake-unit]').forEach((button) => {
    button.addEventListener('click', () => {
      $('#mistake-dialog').close();
      selectUnit(button.dataset.mistakeUnit);
    });
  });
  $('#mistake-dialog').showModal();
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
