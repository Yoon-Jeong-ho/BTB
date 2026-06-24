(function attachBTBProgress(root) {
  const PROGRESS_KEY = 'btb.study.progress.v1';
  const STATES = ['not_started', 'in_progress', 'done', 'blocked'];
  const STATE_LABELS = {
    not_started: '시작 전',
    in_progress: '진행 중',
    done: '완료',
    blocked: '막힘',
  };

  function defaultProgress() {
    return {
      schemaVersion: 1,
      activeUserId: 'local-default',
      users: {
        'local-default': {
          displayName: '내 로컬 진행',
          lessons: {},
        },
      },
      ui: {
        selectedTrack: '',
        selectedUnit: '',
        filters: { progressState: 'all', query: '' },
      },
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
    if (!parsed.ui) parsed.ui = defaultProgress().ui;
    if (!parsed.ui.filters) parsed.ui.filters = { progressState: 'all', query: '' };
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
      store.users[resolvedUserId] = { displayName: '새 로컬 진행', lessons: {} };
    }
    if (!store.users[resolvedUserId].lessons) store.users[resolvedUserId].lessons = {};
    return store.users[resolvedUserId];
  }

  function lessonState(store, userId, unitPath) {
    return ensureUser(store, userId).lessons[unitPath] || {
      state: 'not_started',
      percent: 0,
      checkpoints: {},
      note: '',
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
    store.ui.selectedUnit = unitPath;
    store.ui.selectedTrack = unitPath.split('/')[0];
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
    defaultProgress,
    loadProgress,
    saveProgress,
    ensureUser,
    lessonState,
    upsertLessonProgress,
    mergeImportedProgress,
    createMemoryStorage,
  };

  root.BTBProgress = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
