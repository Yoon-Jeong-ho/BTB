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
  if (label === '단원 안내') return '목표와 학습 순서';
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
        <div class="start-callout">처음이라면 단원 안내와 핵심 이론으로 목표를 잡고, 코드 읽기 → 실행 → 결과 해석 → 메모 순서로 진행하세요.</div>
        ${prerequisiteReadinessFor(unit)}
        <h3>학습 순서</h3>
        <ol class="learning-steps">
          ${learningStepsFor(unit).map((step) => `<li><strong>${escapeHtml(step.label)}</strong><span>${escapeHtml(step.description)}</span></li>`).join('')}
        </ol>
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
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(sectionLabel)}</span><span class="source-badge">${escapeHtml(documentSourceLabel(section))}</span></div>${renderCodeExplanation(section, text)}<pre class="code-block"><code>${escapeHtml(annotateCodeWithInlineHints(section, text))}</code></pre>${renderRunPanel(section, unit, text)}`;
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
  if (!isRunnableCodeSection(section)) return '';
  const plan = runPlanFor(section, unit);
  const symbols = extractPythonSymbols(source).map((symbol) => symbol.replace(/\(\)$/, ''));
  return `<section class="run-panel" aria-label="Python 코드 실행">
    <div>
      <p class="eyebrow">읽은 뒤 실행</p>
      <h4>이 코드를 내 환경에서 확인하기</h4>
      <p>위 코드를 먼저 훑은 다음 실행해 보세요. 종료 코드, 선택된 CPU/GPU, 출력과 산출물이 아래에 정리됩니다.</p>
    </div>
    <div class="run-actions">
      <button type="button" data-run-code data-run-path="${escapeHtml(cleanHref(section.href))}">${escapeHtml(displaySectionLabel(section))} 실행</button>
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

function isRunnableCodeSection(section) {
  return section.type === 'code' && /(?:scratch_lab|framework_lab|analysis|run_stage)\.py$/.test(cleanHref(section.href));
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
  const path = cleanHref(section.href);
  const declared = (unit?.required_outputs || []).filter((item) => !/^runnable README|theory note|prerequisite checklist$/i.test(item));
  if (path.endsWith('run_stage.py')) return declared.length ? declared : ['artifacts/<timestamp>/metrics.json', 'figures/', 'predictions/', 'summary.md'];
  if (path.endsWith('analysis.py')) return ['해석 노트', '관찰 지표', ...declared.filter((item) => /analysis|report|observed/i.test(item))].slice(0, 4);
  if (path.endsWith('framework_lab.py')) return declared.filter((item) => /framework|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['프레임워크 실습 요약']).slice(0, 4);
  if (path.endsWith('scratch_lab.py')) return declared.filter((item) => /scratch|figure|svg|metrics/i.test(item)).slice(0, 4).concat(['기초 실습 요약']).slice(0, 4);
  return declared.length ? declared.slice(0, 4) : ['지표 파일', 'figure 또는 markdown report'];
}

function importantNumbersForRun(section, unit) {
  const path = cleanHref(section.href);
  const terms = unit?.key_terms || [];
  if (path.endsWith('run_stage.py')) return ['주요 평가 지표', '기준 모델 대비 좋은 모델', '학습/평가 데이터 수'];
  if (path.endsWith('analysis.py')) return ['빠진 결과물 수', '실패 사례 수', '해석 노트가 강조한 핵심 지표'];
  if (path.endsWith('framework_lab.py')) return ['loss 또는 accuracy 추세', '기초 실습과 같은 모양/지표인지', '실행 장치와 재실행 기준값'];
  if (path.endsWith('scratch_lab.py')) return ['입력/출력 모양', '핵심 계산 결과', terms[0] ? `${terms[0]} 관측값` : '작은 예제 지표'];
  return ['종료 코드', '지표', '결과물 위치'];
}

function goodOutcomeForRun(section, unit) {
  const path = cleanHref(section.href);
  const deterministic = unit?.deterministic ? ' 같은 설정으로 재실행해도 핵심 숫자가 유지되어야 합니다.' : '';
  if (path.endsWith('run_stage.py')) return `종료 코드 0, 지표·그림·예측 샘플이 생기고 단원 안내의 기준 모델 질문에 답할 수 있으면 좋습니다.${deterministic}`;
  if (path.endsWith('analysis.py')) return `기초 실습 코드와 프레임워크 실습 코드를 먼저 실행한 뒤, 이전 실행 결과물을 빠짐없이 읽고 해석 노트에 실패 사례와 다음 실험 질문이 남으면 좋습니다.${deterministic}`;
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
  const path = cleanHref(section.href);
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
  const path = cleanHref(section.href);
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
  if (normalized.includes('forward')) return 'tensor 입력이 logit·embedding·action 같은 모델 출력으로 바뀌는 계산 경로입니다.';
  if (normalized.includes('train')) return 'batch → loss → optimizer step이 연결되는 학습 루프입니다.';
  if (normalized.includes('evaluate') || normalized.includes('metric') || normalized.includes('score')) return '단원에서 비교할 지표를 계산하므로 단원 안내의 성공 기준과 나란히 확인하세요.';
  if (normalized.includes('compute') || normalized.includes('calculate')) return '중간 텐서나 수치를 최종 지표로 바꾸는 계산입니다.';
  if (normalized.includes('build') || normalized.includes('create') || normalized.includes('prepare') || normalized.includes('make')) return '작은 데이터, 모델, 설정 중 무엇을 고정해 비교 조건을 만드는지 확인하세요.';
  if (normalized.includes('generate') || normalized.includes('sample') || normalized.includes('decode')) return '모델 출력이 사람이 읽을 수 있는 토큰·설명·행동으로 바뀌는 지점입니다.';
  if (normalized.includes('write') || normalized.includes('save')) return '브라우저와 해석 노트가 다시 볼 결과물을 저장하는 지점입니다.';
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
    detail = checkpointActionDetail(nextCheckpoint);
  }
  if (!nextCheckpoint && answered < quizItems.length && selfCheckStats.done > 0) {
    action = '미니 퀴즈 풀기';
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

function checkpointActionDetail(checkpoint) {
  const normalized = String(checkpoint || '').toLowerCase();
  if (['readme', 'theory', 'prereqs'].includes(normalized)) return '코드를 돌리기 전에 왜 배우는지, 어떤 선행 개념이 필요한지 먼저 확인하세요.';
  if (normalized.includes('scratch') || normalized.includes('framework') || normalized.includes('실행 명령')) return '실행 후 산출물 뷰어에서 이번 실행이 만든 지표와 그림을 확인하세요.';
  if (normalized.includes('analysis')) return '숫자를 결론으로 바꾸고, 실패 사례와 다음 실험 질문을 남기세요.';
  if (normalized.includes('reflection')) return '헷갈린 개념, 오답 이유, 다음 단원에서 확인할 질문을 한 줄 이상 남기세요.';
  return '완료하지 않은 체크포인트를 하나씩 닫으면 다음 행동이 갱신됩니다.';
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

function quizForUnit(unit) {
  const keyTerms = unit.key_terms || [];
  const outputs = displayOutputList((unit.required_outputs || []).filter((item) => !/^runnable README|theory note|prerequisite checklist$/i.test(item)));
  const analysis = unit.analysis_questions || [];
  return [
    {
      id: 'goal',
      type: 'single',
      prompt: '이 단원의 가장 중요한 학습 목표는 무엇인가요?',
      explanation: `정답은 단원 목표와 직접 연결됩니다: ${unit.objective || '단원 안내의 첫 설명을 자기 말로 바꾸는 것'}`,
      options: [
        { id: 'objective', label: unit.objective || '단원 안내와 핵심 이론의 목표를 코드와 연결해 설명한다.', correct: true, explain: '단원 목표를 먼저 잡아야 실행 결과를 해석할 수 있습니다.' },
        { id: 'skip', label: '일단 모든 파일을 순서 없이 실행한다.', correct: false, explain: '실행은 중요하지만 목표 없이 돌리면 숫자의 의미를 놓치기 쉽습니다.' },
        { id: 'memorize', label: '용어를 영어 이름 그대로 외운다.', correct: false, explain: '암기보다 입력 모양, 지표, 결과물로 확인하는 것이 BTB의 흐름입니다.' },
      ],
    },
    {
      id: 'artifacts',
      type: 'multi',
      prompt: '실행 후 확인해야 할 산출물을 고르세요.',
      explanation: '결과물은 학습의 증거입니다. 위치·숫자·그림을 함께 확인해야 다음 분석 질문에 답할 수 있습니다.',
      options: [
        { id: 'expected-a', label: outputs[0] || '지표 파일', correct: true, explain: '지표는 이번 실행을 비교할 기준입니다.' },
        { id: 'expected-b', label: outputs[1] || '해석 노트 또는 그림', correct: true, explain: '그림/분석 문서는 숫자를 사람이 읽는 결론으로 바꿉니다.' },
        { id: 'terminal-only', label: '터미널 글자만 보고 닫기', correct: false, explain: '원문 로그만 보면 다시 확인할 수 있는 근거가 남지 않습니다.' },
      ],
    },
    {
      id: 'concept',
      type: 'short',
      prompt: `${keyTerms[0] || '핵심 용어'}를 자기 말로 한 문장으로 설명해 보세요.`,
      expected: analysis[0] || `${keyTerms[0] || '핵심 개념'}이 실행 결과와 어떻게 연결되는지 설명`,
      explanation: '짧은 답변은 자동 정답 하나로 고정하지 않습니다. 자기 말 설명을 남기고, 아래 기준과 비교하세요.',
      options: [],
    },
  ];
}

function renderQuizPanel(unit, quizItems, quizAnswers, wrongNotes) {
  const answered = quizItems.filter((item) => quizAnswers[item.id]).length;
  return `<section class="quiz-panel" aria-label="미니 퀴즈">
    <h3>미니 퀴즈 <span>${answered}/${quizItems.length} 완료</span></h3>
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
