(function attachBTBProgress(root) {
  const PROGRESS_KEY = 'btb.study.progress.v1';
  const STATES = ['not_started', 'in_progress', 'done', 'blocked'];
  const STATE_LABELS = {
    not_started: '시작 전',
    in_progress: '진행 중',
    done: '완료',
    blocked: '질문 필요',
  };

  function defaultUI() {
    return {
      selectedTrack: '',
      selectedUnit: '',
      selectedRoute: 'full',
      filters: { progressState: 'all', query: '' },
    };
  }

  function defaultProgress() {
    return {
      schemaVersion: 1,
      activeUserId: 'local-default',
      users: {
        'local-default': {
          displayName: '내 학습 기록',
          lessons: {},
          ui: defaultUI(),
        },
      },
      ui: defaultUI(),
    };
  }

  function ensureValidStore(parsed) {
    if (!parsed || parsed.schemaVersion !== 1 || !parsed.users || !parsed.activeUserId) {
      throw new Error('unsupported progress schema');
    }
    if (!parsed.users[parsed.activeUserId]) {
      const firstUser = Object.keys(parsed.users)[0];
      if (!firstUser) throw new Error('progress store has no users');
      parsed.activeUserId = firstUser;
    }
    if (!parsed.ui) parsed.ui = defaultUI();
    if (!parsed.ui.selectedRoute) parsed.ui.selectedRoute = 'full';
    if (!parsed.ui.filters) parsed.ui.filters = { progressState: 'all', query: '' };
    for (const user of Object.values(parsed.users)) {
      if (!user.lessons) user.lessons = {};
      if (!user.ui) user.ui = { ...defaultUI(), ...parsed.ui, filters: { ...defaultUI().filters, ...(parsed.ui.filters || {}) } };
      if (!user.ui.selectedRoute) user.ui.selectedRoute = parsed.ui.selectedRoute || 'full';
      if (!user.ui.filters) user.ui.filters = { progressState: 'all', query: '' };
    }
    return parsed;
  }

  function loadProgress(storage) {
    storage = storage || root.localStorage;
    try {
      const raw = storage.getItem(PROGRESS_KEY);
      if (!raw) return defaultProgress();
      return ensureValidStore(JSON.parse(raw));
    } catch (error) {
      try {
        const corruptValue = storage.getItem(PROGRESS_KEY) || '';
        storage.setItem(`${PROGRESS_KEY}.corrupt.${Date.now()}`, corruptValue);
      } catch (_) {
        // Browsing still works when storage is unavailable.
      }
      return defaultProgress();
    }
  }

  function saveProgress(store, storage) {
    storage = storage || root.localStorage;
    store.updatedAt = new Date().toISOString();
    storage.setItem(PROGRESS_KEY, JSON.stringify(store));
    return store;
  }

  function ensureUser(store, userId) {
    const resolvedUserId = userId || store.activeUserId || 'local-default';
    if (!store.users[resolvedUserId]) {
      store.users[resolvedUserId] = { displayName: '새 학습 기록', lessons: {}, ui: defaultUI() };
    }
    if (!store.users[resolvedUserId].lessons) store.users[resolvedUserId].lessons = {};
    if (!store.users[resolvedUserId].ui) store.users[resolvedUserId].ui = defaultUI();
    if (!store.users[resolvedUserId].ui.filters) store.users[resolvedUserId].ui.filters = { progressState: 'all', query: '' };
    return store.users[resolvedUserId];
  }

  function userUI(store, userId) {
    return ensureUser(store, userId).ui;
  }

  function updateUserUI(store, userId, patch) {
    const resolvedUserId = userId || store.activeUserId || 'local-default';
    const ui = userUI(store, resolvedUserId);
    const nextFilters = patch.filters ? { ...ui.filters, ...patch.filters } : ui.filters;
    ensureUser(store, resolvedUserId).ui = { ...ui, ...patch, filters: nextFilters };
    if (store.activeUserId === resolvedUserId) {
      store.ui = { ...ensureUser(store, resolvedUserId).ui, filters: { ...ensureUser(store, resolvedUserId).ui.filters } };
    }
    return store;
  }

  function lessonState(store, userId, unitPath) {
    return ensureUser(store, userId).lessons[unitPath] || {
      state: 'not_started',
      percent: 0,
      checkpoints: {},
      selfChecks: {},
      quizAnswers: {},
      wrongNotes: {},
      runEvidence: null,
      note: '',
    };
  }

  function hasSubstantiveAnswer(value) {
    if (Array.isArray(value)) return value.some((item) => String(item || '').trim());
    return Boolean(String(value || '').trim());
  }

  function runEvidenceVerified(evidence) {
    if (!evidence || evidence.returncode !== 0) return false;
    if (!Array.isArray(evidence.artifactNames) || evidence.artifactNames.length === 0) return false;
    if (evidence.device === 'cuda' && evidence.artifactDevice !== 'cuda') return false;
    if (evidence.artifactDevice && evidence.device && evidence.artifactDevice !== evidence.device) return false;
    return true;
  }

  function masteryEvidence(lesson, requiredCheckpoints, requiredQuizIds) {
    const progress = lesson || {};
    const checkpoints = requiredCheckpoints || [];
    const quizIds = Array.isArray(requiredQuizIds) ? requiredQuizIds : [];
    const quizAnswers = progress.quizAnswers || {};
    const items = [
      {
        key: 'readings',
        label: '필수 읽기와 실습 체크',
        done: checkpoints.length > 0 && checkpoints.every((checkpoint) => progress.checkpoints?.[checkpoint]),
      },
      {
        key: 'run',
        label: '성공한 실행 증거',
        done: runEvidenceVerified(progress.runEvidence),
      },
      {
        key: 'quiz',
        label: '단원 퀴즈 답변',
        done: quizIds.length > 0 && quizIds.every((id) => hasSubstantiveAnswer(quizAnswers[id]?.answer)),
      },
      {
        key: 'reflection',
        label: '회고 또는 다음 가설',
        done: hasSubstantiveAnswer(progress.note),
      },
    ];
    return {
      items,
      done: items.filter((item) => item.done).length,
      total: items.length,
      verified: items.every((item) => item.done),
    };
  }

  function upsertLessonProgress(store, userId, unitPath, patch, now) {
    const timestamp = now || new Date().toISOString();
    const user = ensureUser(store, userId);
    const previous = lessonState(store, userId, unitPath);
    user.lessons[unitPath] = {
      ...previous,
      ...patch,
      updatedAt: timestamp,
      lastOpenedAt: timestamp,
    };
    store.activeUserId = userId;
    updateUserUI(store, userId, { selectedUnit: unitPath, selectedTrack: unitPath.split('/')[0] });
    return store;
  }

  function mergeImportedProgress(store, incoming) {
    ensureValidStore(incoming);
    store.users = { ...store.users, ...incoming.users };
    if (incoming.activeUserId && store.users[incoming.activeUserId]) {
      store.activeUserId = incoming.activeUserId;
    }
    if (incoming.ui) {
      store.ui = { ...store.ui, ...incoming.ui, filters: { ...store.ui.filters, ...(incoming.ui.filters || {}) } };
    }
    for (const userId of Object.keys(store.users)) {
      ensureUser(store, userId);
    }
    if (store.users[store.activeUserId]) {
      store.ui = { ...userUI(store, store.activeUserId), filters: { ...userUI(store, store.activeUserId).filters } };
    }
    return store;
  }

  function createMemoryStorage(seed) {
    const map = new Map(Object.entries(seed || {}));
    return {
      getItem(key) {
        return map.has(key) ? map.get(key) : null;
      },
      setItem(key, value) {
        map.set(key, String(value));
      },
      removeItem(key) {
        map.delete(key);
      },
      keys() {
        return Array.from(map.keys());
      },
    };
  }

  const api = {
    PROGRESS_KEY,
    STATES,
    STATE_LABELS,
    defaultUI,
    defaultProgress,
    loadProgress,
    saveProgress,
    ensureUser,
    userUI,
    updateUserUI,
    lessonState,
    hasSubstantiveAnswer,
    runEvidenceVerified,
    masteryEvidence,
    upsertLessonProgress,
    mergeImportedProgress,
    createMemoryStorage,
  };

  root.BTBProgress = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
