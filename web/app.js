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

init();

async function init() {
  try {
    const response = await fetch('catalog.json');
    if (!response.ok) throw new Error(`catalog load failed: ${response.status}`);
    catalog = await response.json();
  } catch (error) {
    detail.innerHTML = `<p class="empty">catalog.json을 읽지 못했습니다. 저장소 루트에서 <code>python -m http.server 8000</code>를 실행한 뒤 <code>http://localhost:8000/web/</code>을 열어 주세요.</p>`;
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
}

function renderProfiles() {
  profileSelect.innerHTML = Object.entries(progressStore.users)
    .map(([id, user]) => `<option value="${escapeHtml(id)}">${escapeHtml(user.displayName || id)}</option>`)
    .join('');
  profileSelect.value = activeUserId;
}

function renderTracks() {
  trackList.innerHTML = catalog.tracks.map((track) => {
    const stats = trackStats(track);
    return `<button class="track-card" type="button" aria-pressed="${track.id === selectedTrackId}" data-track="${escapeHtml(track.id)}">
      <div class="track-top"><span class="track-title">${escapeHtml(track.title)}</span><span class="track-meta">${stats.done}/${stats.total}</span></div>
      <div class="track-meta">${escapeHtml(track.id)}</div>
      <p>${escapeHtml(track.summary || '이 트랙의 README에서 학습 방향을 확인한다.')}</p>
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
    <p>${escapeHtml(unit.objective || '목표 설명은 사이트 안의 README 탭에서 확인하세요.')}</p>
    <div class="chips">${outputChips}</div>
  </button>`;
}

function renderDetail() {
  const unit = findUnit(selectedUnitPath);
  if (!unit) {
    detail.innerHTML = '<p class="empty">단원을 선택하면 목표, 실험 산출물, 체크리스트, README/THEORY/실습 코드가 사이트 안에서 표시됩니다.</p>';
    return;
  }
  const progress = lessonState(unit.path);
  const checkpoints = unit.checkpoints.length ? unit.checkpoints : ['README'];
  const checked = progress.checkpoints || {};
  const percent = completionPercent(checkpoints, checked, progress.state);
  const sections = lessonSectionsFor(unit);
  const selectedSection = sections.find((section) => hrefEquals(section.href, selectedResourceHref)) || sections[0];
  selectedResourceHref = selectedSection.href;

  detail.innerHTML = `<section class="lesson-hero">
      <div>
        <h2 id="detail-title">${escapeHtml(unit.title)}</h2>
        <p class="unit-meta">${escapeHtml(unit.path)} · curriculum: ${escapeHtml(unit.status)} · personal: ${STATE_LABELS[progress.state]}</p>
        <p>${escapeHtml(unit.objective || '')}</p>
      </div>
      <div>
        <div class="status-buttons" aria-label="진행 상태 변경">
          ${STATES.map((state) => `<button type="button" data-state="${state}" class="${state === progress.state ? 'active' : ''}">${STATE_LABELS[state]}</button>`).join('')}
        </div>
        <div class="progress-bar" aria-label="체크리스트 ${percent}% 완료"><span style="width:${percent}%"></span></div>
      </div>
    </section>
    <div class="lesson-workspace">
      <aside class="lesson-guide" aria-label="학습 진행 가이드">
        <div class="start-callout">처음이라면 여기서 시작하세요: README와 THEORY를 사이트 안에서 읽고 scratch → framework → analysis → reflection 순서로 진행합니다.</div>
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
        <h3>로컬 메모</h3>
        <textarea class="notes" id="unit-note" placeholder="이 메모는 현재 브라우저 localStorage에만 저장됩니다.">${escapeHtml(progress.note || '')}</textarea>
      </aside>
      <section class="lesson-reader" aria-live="polite">
        <div class="reader-header">
          <div>
            <p class="eyebrow">학습 자료</p>
            <h3>사이트 안에서 읽고 실습 흐름으로 넘어가기</h3>
          </div>
          <button id="mark-section-complete" type="button">현재 자료 체크</button>
        </div>
        <div class="document-tabs" role="tablist" aria-label="단원 자료">
          ${sections.map((section) => `<button type="button" role="tab" data-section-href="${escapeHtml(section.href)}" aria-selected="${hrefEquals(section.href, selectedSection.href)}">${escapeHtml(section.label)}</button>`).join('')}
        </div>
        <article id="lesson-content" class="lesson-content"><p class="empty">자료를 불러오는 중입니다.</p></article>
      </section>
    </div>`;

  bindDetailEvents(unit, checkpoints, checked, progress.state, selectedSection);
  loadLessonSection(unit, selectedSection);
}

function bindDetailEvents(unit, checkpoints, checked, currentState, selectedSection) {
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
      content.innerHTML = `<div class="document-title"><span>${escapeHtml(section.label)}</span><code>${escapeHtml(cleanHref(section.href))}</code></div><pre class="code-block"><code>${escapeHtml(text)}</code></pre>`;
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

function updateMarkSectionButton(unit, section, button) {
  if (!button) return;
  if (!section.checkpoint) {
    button.disabled = true;
    button.textContent = '참고 자료';
    return;
  }
  const checked = lessonState(unit.path).checkpoints || {};
  button.disabled = Boolean(checked[section.checkpoint]);
  button.textContent = checked[section.checkpoint] ? '체크 완료' : `${section.checkpoint} 체크`;
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
  const units = catalog.tracks.flatMap((track) => track.units);
  return units.find((unit) => lessonState(unit.path).state !== 'done') || units[0] || null;
}

function learningStepsFor(unit) {
  return [
    { label: '이론 읽기', description: 'README / THEORY / PREREQS로 왜 배우는지와 선행 개념을 잡는다.' },
    { label: 'scratch 실행', description: 'scratch_lab.py로 작은 수치와 직접 계산을 확인한다.' },
    { label: 'framework 실행', description: 'framework_lab.py로 PyTorch나 프레임워크 관측을 비교한다.' },
    { label: 'analysis 정리', description: 'analysis.py와 analysis.md로 관측값을 한국어 해석으로 남긴다.' },
    { label: 'reflection 작성', description: 'reflection.md에 헷갈린 점, 실패 사례, 다음 질문을 적는다.' },
  ].filter((step) => {
    if (step.label.includes('scratch')) return unit.checkpoints.includes('scratch lab');
    if (step.label.includes('framework')) return unit.checkpoints.includes('framework lab');
    if (step.label.includes('analysis')) return unit.checkpoints.includes('analysis script') || unit.checkpoints.includes('analysis note');
    if (step.label.includes('reflection')) return unit.checkpoints.includes('reflection');
    return true;
  });
}

function studyLinksFor(unit) {
  const links = [
    { href: '../docs/02_study_guide.md', label: 'Study guide', reason: '무기초 → LLM/RLHF/Multimodal/VLA 경로 확인' },
  ];
  if (unit.path.includes('05_advanced_nlp_llm/06_rlhf')) {
    links.push({ href: '../docs/05_rl_primer_for_rlhf.md', label: 'RL primer for RLHF', reason: 'reward/policy/rollout/KL/PPO 선행 정리' });
  }
  if (unit.path.startsWith('10_vla/')) {
    links.push({ href: '../09_multimodal/README.md', label: '09 Multimodal recap', reason: 'VQA에서 action grounding으로 넘어가기 전 복습' });
  }
  const trackReadme = `../${unit.path.split('/')[0]}/README.md`;
  links.push({ href: trackReadme, label: 'Track README', reason: '현재 트랙의 역할과 다음 연결 확인' });
  return links;
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
  const units = catalog.tracks.flatMap((track) => track.units);
  const done = units.filter((unit) => lessonState(unit.path).state === 'done').length;
  const profileName = currentUser().displayName || activeUserId;
  $('#overall-progress').textContent = units.length ? `${Math.round((done / units.length) * 100)}% · ${profileName}` : `0% · ${profileName}`;
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
    alert('가져오기 실패: BTB progress JSON 형식을 확인하세요.');
  }
}

function resetProgress() {
  if (!confirm('현재 브라우저의 BTB 로컬 진행률을 삭제할까요? GitHub 데이터는 바뀌지 않습니다.')) return;
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
