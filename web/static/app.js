/**
 * Footy Predictor v13 Oracle — Alpine.js SPA
 * 50-expert ensemble · 56.2% accuracy · Self-learning · Timeout handling
 * Theme toggle · Keyboard shortcuts · Loading states · Auto-refresh
 */

const REQUEST_TIMEOUT_MS = 15000;
const MAX_RETRIES = 3;
const RETRY_BACKOFF_MS = 1000;
const ITEMS_PER_PAGE = 50;
const AUTO_REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

document.addEventListener('alpine:init', () => {

  Alpine.data('app', () => ({
    // ── routing ──
    view: 'main',
    tab: 'matches',
    matchId: null,

    // ── main view state ──
    matches: [],
    loading: true,
    league: 'all',
    valueBets: [],
    loadingBets: false,
    stats: null,
    loadingStats: false,
    performance: null,
    loadingPerf: false,
    lastUpdated: null,
    days: 14,
    model: 'v13_oracle',
    sortMode: 'kickoff',
    searchQuery: '',

    // ── pagination state ──
    matchesPage: 1,
    matchesLoadingMore: false,

    // ── league table state ──
    tableComp: 'PL',
    tableCompetitions: ['PL', 'PD', 'SA', 'BL1', 'FL1'],
    leagueTable: [],
    loadingTable: false,

    // ── BTTS & O/U state ──
    bttsOu: null,
    loadingBtts: false,

    // ── Accumulators state ──
    accumulators: [],
    loadingAccas: false,

    // ── League Form Table state ──
    formTableComp: 'PL',
    formTable: [],
    loadingFormTable: false,

    // ── Accuracy Dashboard state ──
    accuracyStats: null,
    loadingAccuracy: false,
    accuracyDays: 30,

    // ── Round Preview state ──
    roundPreview: null,
    loadingRoundPreview: false,
    roundPreviewComp: 'PL',

    // ── Post-Match Review state ──
    postMatchReview: null,
    loadingReview: false,

    // ── Training state ──
    trainingStatus: null,
    loadingTraining: false,

    // ── Model lab state ──
    modelLab: null,
    loadingModelLab: false,

    // ── Brain (self-learning) state ──
    brainStatus: null,
    loadingBrain: false,

    // ── Sources state ──
    sourcesData: null,

    // ── Season Simulation state ──
    simComp: 'PL',
    simData: null,
    loadingSim: false,

    // ── Team Profile state ──
    teamProfile: null,
    loadingProfile: false,
    profileTeamName: '',

    // ── Streaks state ──
    streaksData: null,
    loadingStreaks: false,

    // ── Prediction History state ──
    predHistory: null,
    loadingHistory: false,

    // ── match detail state ──
    md: null,
    loadingDetail: false,
    matchExperts: null,
    loadingExperts: false,
    matchH2H: null,
    loadingH2H: false,
    matchForm: null,
    matchNarrative: null,
    loadingAI: false,
    matchXG: null,
    loadingXG: false,
    matchPatterns: null,
    loadingPatterns: false,

    // ── multi-model state ──
    multiModelData: null,
    loadingMultiModel: false,

    // ── UI state ──
    theme: 'dark',
    showKeyboardHelp: false,
    showComparisonModal: false,
    comparisonMatches: [],
    autoRefreshEnabled: false,
    showInsightsDropdown: false,
    showAnalysisDropdown: false,
    showSystemDropdown: false,

    // ── request lifecycle ──
    _matchAbort: null,
    _matchRequestId: 0,
    _matchesAbortController: null,
    _matchAbortController: null,
    _mainTabs: ['matches', 'insights', 'btts', 'accas', 'form', 'table', 'accuracy', 'review', 'simulation', 'streaks', 'history', 'stats', 'training', 'lab', 'brain', 'sources'],
    _errorToast: null,
    _errorTimeout: null,
    _successToast: null,
    _successTimeout: null,
    _autoRefreshTimer: null,
    _autoRefreshFailures: 0,
    _retryMap: new Map(),
    _searchTimeout: null,
    _debouncedSearchQuery: '',

    // ── helpers ──
    async _fetch(url, opts = {}) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      try {
        const signal = opts.signal || controller.signal;
        const requestSignal = new AbortController();
        const finalSignal = signal.aborted ? requestSignal.signal : signal;

        const r = await fetch(url, { ...opts, signal: finalSignal });
        clearTimeout(timeoutId);

        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          const msg = body.error || `Request failed (${r.status})`;
          this._showError(msg);
          throw new Error(msg);
        }
        return r.json();
      } catch(e) {
        clearTimeout(timeoutId);
        if (e.name === 'AbortError') throw e;
        throw e;
      }
    },

    async _fetchWithRetry(url, opts = {}, retries = 0) {
      try {
        return await this._fetch(url, opts);
      } catch(e) {
        if (e.name === 'AbortError') throw e;
        if (retries < MAX_RETRIES) {
          const delay = RETRY_BACKOFF_MS * Math.pow(2, retries);
          await new Promise(r => setTimeout(r, delay));
          return this._fetchWithRetry(url, opts, retries + 1);
        }
        throw e;
      }
    },

    _showError(msg) {
      this._errorToast = msg;
      clearTimeout(this._errorTimeout);
      this._errorTimeout = setTimeout(() => { this._errorToast = null; }, 5000);
    },

    _showSuccess(msg) {
      // Success toast via Alpine reactivity — auto-dismiss after 3 seconds
      this._successToast = msg;
      clearTimeout(this._successTimeout);
      this._successTimeout = setTimeout(() => { this._successToast = null; }, 3000);
    },

    _initTheme() {
      try {
        const saved = localStorage.getItem('footy-theme');
        if (saved) {
          this.theme = saved;
        } else {
          this.theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        this._applyTheme();
      } catch (_) { /* ignore */ }
    },

    _applyTheme() {
      if (this.theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      try {
        localStorage.setItem('footy-theme', this.theme);
      } catch (_) { /* ignore */ }
    },

    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark';
      this._applyTheme();
    },

    toggleInsightsDropdown() {
      this.showInsightsDropdown = !this.showInsightsDropdown;
      this.showAnalysisDropdown = false;
      this.showSystemDropdown = false;
    },

    toggleAnalysisDropdown() {
      this.showAnalysisDropdown = !this.showAnalysisDropdown;
      this.showInsightsDropdown = false;
      this.showSystemDropdown = false;
    },

    toggleSystemDropdown() {
      this.showSystemDropdown = !this.showSystemDropdown;
      this.showInsightsDropdown = false;
      this.showAnalysisDropdown = false;
    },

    _initKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        // ? for help
        if (e.key === '?' && !this.showKeyboardHelp) {
          this.showKeyboardHelp = true;
          return;
        }
        // Escape to close modals and dropdowns
        if (e.key === 'Escape') {
          if (this.showKeyboardHelp) this.showKeyboardHelp = false;
          if (this.showComparisonModal) this.showComparisonModal = false;
          if (this.showInsightsDropdown) this.showInsightsDropdown = false;
          if (this.showAnalysisDropdown) this.showAnalysisDropdown = false;
          if (this.showSystemDropdown) this.showSystemDropdown = false;
          return;
        }
        // / for search (focus search input) - only when not in text input
        if (e.key === '/' && e.ctrlKey === false && e.metaKey === false && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
          const searchInput = document.querySelector('.search-input');
          if (searchInput && this.view === 'main' && this.tab === 'matches') {
            e.preventDefault();
            searchInput.focus();
          }
        }
        // Ctrl/Cmd+R to refresh current view
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
          e.preventDefault();
          this.refreshCurrentView();
        }
      });
    },

    _saveUiPrefs() {
      try {
        localStorage.setItem('footy-ui-prefs', JSON.stringify({
          tab: this.tab,
          league: this.league,
          days: this.days,
          sortMode: this.sortMode,
          theme: this.theme,
        }));
      } catch (_) { /* ignore storage issues */ }
    },

    _loadUiPrefs() {
      try {
        const raw = localStorage.getItem('footy-ui-prefs');
        if (!raw) return;
        const prefs = JSON.parse(raw);
        if (prefs.tab) this.tab = prefs.tab;
        if (prefs.league) this.league = prefs.league;
        if (prefs.days) this.days = Number(prefs.days);
        if (prefs.sortMode) this.sortMode = prefs.sortMode;
      } catch (_) { /* ignore invalid persisted prefs */ }
    },

    tabId(tab) { return `tab-${tab}`; },
    panelId(tab) { return `panel-${tab}`; },

    focusTab(tab) {
      requestAnimationFrame(() => document.getElementById(this.tabId(tab))?.focus());
    },

    onTabKeydown(event, currentTab) {
      const tabs = this._mainTabs;
      const idx = tabs.indexOf(currentTab);
      if (idx === -1) return;
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        const next = tabs[(idx + 1) % tabs.length];
        this.switchTab(next);
        this.focusTab(next);
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        const prev = tabs[(idx - 1 + tabs.length) % tabs.length];
        this.switchTab(prev);
        this.focusTab(prev);
      } else if (event.key === 'Home') {
        event.preventDefault();
        this.switchTab(tabs[0]);
        this.focusTab(tabs[0]);
      } else if (event.key === 'End') {
        event.preventDefault();
        const last = tabs[tabs.length - 1];
        this.switchTab(last);
        this.focusTab(last);
      }
    },

    // ── lifecycle ──
    init() {
      this._initTheme();
      this._loadUiPrefs();
      this._initKeyboardShortcuts();

      // Detect URL for routing
      const path = window.location.pathname;
      const matchRoute = path.match(/^\/match\/(\d+)$/);
      if (matchRoute) {
        this.openMatch(matchRoute[1], false);
      } else {
        this.fetchMatches();
      }
      this.fetchLastUpdated();

      // Handle browser back/forward
      window.addEventListener('popstate', (e) => {
        if (e.state?.view === 'match' && e.state?.matchId) {
          this.openMatch(e.state.matchId, false);
        } else {
          this.view = 'main';
          if (!this.matches.length) this.fetchMatches();
        }
      });

      // Theme preference listener
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('footy-theme')) {
          this.theme = e.matches ? 'dark' : 'light';
          this._applyTheme();
        }
      });
    },

    // ═══════ ROUTING ═══════
    goHome() {
      this.view = 'main';
      if (!this.matches.length) this.fetchMatches();
      if (window.location.pathname !== '/') {
        history.pushState({ view: 'main' }, '', '/');
      }
      document.title = 'Footy Predictor';
    },

    async openMatch(matchId, pushState = true) {
      // Cancel any in-flight match requests to prevent stale data overwrites
      if (this._matchAbortController) {
        this._matchAbortController.abort();
      }
      this._matchAbortController = new AbortController();
      const signal = this._matchAbortController.signal;
      const requestId = ++this._matchRequestId;

      this.view = 'match';
      this.matchId = matchId;
      this.md = null;
      this.matchExperts = null;
      this.matchH2H = null;
      this.matchForm = null;
      this.matchNarrative = null;
      this.matchXG = null;
      this.matchPatterns = null;
      this.multiModelData = null;
      this.loadingDetail = true;
      this.loadingExperts = false;
      this.loadingH2H = false;
      this.loadingAI = false;
      this.loadingXG = false;
      this.loadingPatterns = false;
      this.loadingMultiModel = false;

      if (pushState) {
        history.pushState({ view: 'match', matchId }, '', `/match/${matchId}`);
      }
      window.scrollTo(0, 0);

      try {
        this.md = await this._fetchWithRetry(`/api/matches/${matchId}?model=${this.model}`, { signal });
        if (this._matchRequestId !== requestId) return; // stale response
        document.title = `${this.md.home_team} vs ${this.md.away_team} — Footy Predictor`;
      } catch(e) {
        if (e.name === 'AbortError') return;
        console.error(e);
        if (this._matchRequestId === requestId) this.loadingDetail = false;
        return;
      }
      this.loadingDetail = false;

      // Auto-load all match data in parallel
      this._loadMatchExperts(matchId, requestId);
      this._loadMatchH2H(matchId, requestId);
      this._loadMatchForm(matchId, requestId);
      this._loadMatchXG(matchId, requestId);
      this._loadMatchPatterns(matchId, requestId);
    },

    async _loadMatchExperts(matchId, requestId) {
      this.loadingExperts = true;
      try {
        const data = await this._fetchWithRetry(`/api/matches/${matchId}/experts`, { signal: this._matchAbortController?.signal });
        if (this._matchRequestId === requestId) this.matchExperts = data;
      } catch(e) { if (e.name !== 'AbortError') console.error(e); }
      if (this._matchRequestId === requestId) this.loadingExperts = false;
    },

    async _loadMatchH2H(matchId, requestId) {
      this.loadingH2H = true;
      try {
        const data = await this._fetchWithRetry(`/api/matches/${matchId}/h2h`, { signal: this._matchAbortController?.signal });
        if (this._matchRequestId === requestId) this.matchH2H = data;
      } catch(e) { if (e.name !== 'AbortError') console.error(e); }
      if (this._matchRequestId === requestId) this.loadingH2H = false;
    },

    async _loadMatchForm(matchId, requestId) {
      try {
        const data = await this._fetchWithRetry(`/api/matches/${matchId}/form`, { signal: this._matchAbortController?.signal });
        if (this._matchRequestId === requestId) this.matchForm = data;
      } catch(e) { if (e.name !== 'AbortError') console.error(e); }
    },

    async _loadMatchXG(matchId, requestId) {
      this.loadingXG = true;
      try {
        const data = await this._fetchWithRetry(`/api/matches/${matchId}/xg`, { signal: this._matchAbortController?.signal });
        if (this._matchRequestId === requestId) this.matchXG = data;
      } catch(e) { if (e.name !== 'AbortError') console.error(e); }
      if (this._matchRequestId === requestId) this.loadingXG = false;
    },

    async _loadMatchPatterns(matchId, requestId) {
      this.loadingPatterns = true;
      try {
        const data = await this._fetchWithRetry(`/api/matches/${matchId}/patterns`, { signal: this._matchAbortController?.signal });
        if (this._matchRequestId === requestId) this.matchPatterns = data;
      } catch(e) { if (e.name !== 'AbortError') console.error(e); }
      if (this._matchRequestId === requestId) this.loadingPatterns = false;
    },

    async loadMultiModel() {
      if (!this.matchId || this.multiModelData) return;
      this.loadingMultiModel = true;
      try {
        this.multiModelData = await this._fetchWithRetry(`/api/matches/${this.matchId}/models`);
      } catch(e) { console.error(e); }
      this.loadingMultiModel = false;
    },

    async loadAI() {
      if (!this.matchId || this.matchNarrative) return;
      this.loadingAI = true;
      try {
        const d = await this._fetchWithRetry(`/api/matches/${this.matchId}/ai`);
        this.matchNarrative = d.narrative || 'AI analysis unavailable — is Ollama running?';
      } catch(e) { this.matchNarrative = 'Failed to generate analysis.'; }
      this.loadingAI = false;
    },

    // ═══════ API CALLS ═══════
    async fetchMatches() {
      if (this._matchesAbortController) {
        this._matchesAbortController.abort();
      }
      this._matchesAbortController = new AbortController();
      const signal = this._matchesAbortController.signal;
      
      this.loading = true;
      try {
        const d = await this._fetchWithRetry(`/api/matches?days=${this.days}&model=${this.model}`, { signal });
        this.matches = d.matches || [];
        this.matchesPage = 1;
      } catch(e) { 
        if (e.name !== 'AbortError') console.error(e); 
      }
      this.loading = false;
    },

    async loadMoreMatches() {
      if (this.matchesLoadingMore) return;
      this.matchesLoadingMore = true;
      try {
        const d = await this._fetchWithRetry(`/api/matches?days=${this.days}&model=${this.model}&page=${this.matchesPage + 1}`);
        const newMatches = d.matches || [];
        if (newMatches.length > 0) {
          this.matches = [...this.matches, ...newMatches];
          this.matchesPage += 1;
        } else {
          // No more results available, disable button
          this.matchesLoadingMore = false;
          return;
        }
      } catch(e) { console.error(e); }
      this.matchesLoadingMore = false;
    },

    async fetchLastUpdated() {
      try {
        const d = await this._fetchWithRetry('/api/last-updated');
        this.lastUpdated = d.last_updated;
      } catch(e) { /* ignore */ }
    },

    async fetchValueBets() {
      this.loadingBets = true;
      try {
        const d = await this._fetchWithRetry('/api/insights/value-bets?min_edge=0.03');
        this.valueBets = d.bets || [];
      } catch(e) { console.error(e); }
      this.loadingBets = false;
    },

    async fetchBttsOu() {
      this.loadingBtts = true;
      try { this.bttsOu = await this._fetchWithRetry('/api/insights/btts-ou'); } catch(e) { console.error(e); }
      this.loadingBtts = false;
    },

    async fetchAccumulators() {
      this.loadingAccas = true;
      try {
        const d = await this._fetchWithRetry('/api/insights/accumulators');
        this.accumulators = d.accumulators || [];
      } catch(e) { console.error(e); }
      this.loadingAccas = false;
    },

    async fetchFormTable() {
      this.loadingFormTable = true;
      try {
        const d = await this._fetchWithRetry(`/api/insights/form-table/${this.formTableComp}`);
        this.formTable = d.table || [];
      } catch(e) { console.error(e); this.formTable = []; }
      this.loadingFormTable = false;
    },

    selectFormTableComp(c) {
      this.formTableComp = c;
      this.fetchFormTable();
    },

    async fetchAccuracy() {
      this.loadingAccuracy = true;
      try { this.accuracyStats = await this._fetchWithRetry(`/api/insights/accuracy?days_back=${this.accuracyDays}`); } catch(e) { console.error(e); }
      this.loadingAccuracy = false;
    },

    async fetchRoundPreview() {
      this.loadingRoundPreview = true;
      try { this.roundPreview = await this._fetchWithRetry(`/api/insights/round-preview/${this.roundPreviewComp}`); } catch(e) { console.error(e); }
      this.loadingRoundPreview = false;
    },

    selectRoundPreviewComp(c) {
      this.roundPreviewComp = c;
      this.fetchRoundPreview();
    },

    async fetchPostMatchReview() {
      this.loadingReview = true;
      try { this.postMatchReview = await this._fetchWithRetry('/api/insights/post-match-review?days_back=7'); } catch(e) { console.error(e); }
      this.loadingReview = false;
    },

    async fetchTrainingStatus() {
      this.loadingTraining = true;
      try { this.trainingStatus = await this._fetchWithRetry('/api/training/status'); } catch(e) { console.error(e); }
      this.loadingTraining = false;
    },

    async fetchModelLab() {
      this.loadingModelLab = true;
      try { this.modelLab = await this._fetchWithRetry('/api/model-lab'); } catch(e) { console.error(e); }
      this.loadingModelLab = false;
    },

    async fetchBrainStatus() {
      this.loadingBrain = true;
      try { this.brainStatus = await this._fetchWithRetry('/api/self-learning/status'); } catch(e) { console.error(e); }
      this.loadingBrain = false;
    },

    async fetchSourcesData() {
      try { this.sourcesData = await this._fetchWithRetry('/api/sources'); } catch(e) { console.error(e); }
    },

    async fetchSeasonSim() {
      this.loadingSim = true;
      try { this.simData = await this._fetchWithRetry(`/api/season-simulation/${this.simComp}`); } catch(e) { console.error(e); }
      this.loadingSim = false;
    },

    selectSimComp(c) {
      this.simComp = c;
      this.simData = null;
      this.fetchSeasonSim();
    },

    async fetchTeamProfile(teamName) {
      this.loadingProfile = true;
      this.profileTeamName = teamName;
      try {
        this.teamProfile = await this._fetchWithRetry(`/api/team/${encodeURIComponent(teamName)}/profile`);
      } catch(e) { console.error(e); this.teamProfile = null; }
      this.loadingProfile = false;
    },

    async fetchStreaks() {
      this.loadingStreaks = true;
      try { this.streaksData = await this._fetchWithRetry('/api/streaks'); } catch(e) { console.error(e); }
      this.loadingStreaks = false;
    },

    async fetchPredHistory() {
      this.loadingHistory = true;
      try { this.predHistory = await this._fetchWithRetry('/api/predictions/history'); } catch(e) { console.error(e); }
      this.loadingHistory = false;
    },

    async triggerRefresh() {
      try {
        await this._fetch('/api/refresh', { method: 'POST' });
        this._showSuccess('Data refresh started');
      } catch(e) {
        this._showError('Failed to start refresh');
      }
    },

    async fetchStats() {
      this.loadingStats = true;
      try { this.stats = await this._fetchWithRetry('/api/stats'); } catch(e) { console.error(e); }
      this.loadingStats = false;
    },

    async fetchPerformance() {
      this.loadingPerf = true;
      try { this.performance = await this._fetchWithRetry(`/api/performance?model=${this.model}`); } catch(e) { console.error(e); }
      this.loadingPerf = false;
    },

    async fetchLeagueTable() {
      this.loadingTable = true;
      try {
        const d = await this._fetchWithRetry(`/api/league-table/${this.tableComp}`);
        this.leagueTable = d.standings || [];
      } catch(e) { console.error(e); this.leagueTable = []; }
      this.loadingTable = false;
    },

    async refreshCurrentView() {
      if (this.view === 'match' && this.matchId) {
        await this.openMatch(this.matchId, false);
        return;
      }
      if (this.tab === 'matches') return this.fetchMatches();
      if (this.tab === 'insights') return this.fetchValueBets();
      if (this.tab === 'btts') return this.fetchBttsOu();
      if (this.tab === 'accas') return this.fetchAccumulators();
      if (this.tab === 'form') return this.fetchFormTable();
      if (this.tab === 'accuracy') return this.fetchAccuracy();
      if (this.tab === 'review') {
        await this.fetchPostMatchReview();
        return this.fetchRoundPreview();
      }
      if (this.tab === 'stats') {
        await this.fetchStats();
        return this.fetchPerformance();
      }
      if (this.tab === 'table') return this.fetchLeagueTable();
      if (this.tab === 'training') return this.fetchTrainingStatus();
      if (this.tab === 'lab') return this.fetchModelLab();
      if (this.tab === 'brain') return this.fetchBrainStatus();
      if (this.tab === 'sources') return this.fetchSourcesData();
      if (this.tab === 'simulation') return this.fetchSeasonSim();
      if (this.tab === 'streaks') return this.fetchStreaks();
      if (this.tab === 'history') return this.fetchPredHistory();
    },

    selectTableComp(c) {
      this.tableComp = c;
      this.fetchLeagueTable();
    },

    switchTab(t) {
      this.tab = t;
      this.showInsightsDropdown = false;
      this.showAnalysisDropdown = false;
      this.showSystemDropdown = false;
      this._saveUiPrefs();
      if (t === 'insights' && !this.valueBets.length) this.fetchValueBets();
      if (t === 'btts' && !this.bttsOu) this.fetchBttsOu();
      if (t === 'accas' && !this.accumulators.length) this.fetchAccumulators();
      if (t === 'form' && !this.formTable.length) this.fetchFormTable();
      if (t === 'accuracy' && !this.accuracyStats) this.fetchAccuracy();
      if (t === 'review' && !this.postMatchReview) { this.fetchPostMatchReview(); this.fetchRoundPreview(); }
      if (t === 'stats' && !this.stats) { this.fetchStats(); this.fetchPerformance(); }
      if (t === 'table' && !this.leagueTable.length) this.fetchLeagueTable();
      if (t === 'training' && !this.trainingStatus) this.fetchTrainingStatus();
      if (t === 'lab' && !this.modelLab) this.fetchModelLab();
      if (t === 'brain' && !this.brainStatus) this.fetchBrainStatus();
      if (t === 'sources' && !this.sourcesData) this.fetchSourcesData();
      if (t === 'simulation' && !this.simData) this.fetchSeasonSim();
      if (t === 'streaks' && !this.streaksData) this.fetchStreaks();
      if (t === 'history' && !this.predHistory) this.fetchPredHistory();
    },

    updateSearchQuery(query) {
      clearTimeout(this._searchTimeout);
      this._searchTimeout = setTimeout(() => {
        this._debouncedSearchQuery = query;
      }, 300);
    },

    toggleAutoRefresh() {
      this.autoRefreshEnabled = !this.autoRefreshEnabled;
      if (this.autoRefreshEnabled) {
        this._startAutoRefresh();
      } else {
        this._stopAutoRefresh();
      }
    },

    _startAutoRefresh() {
      if (this._autoRefreshTimer) clearInterval(this._autoRefreshTimer);
      this._autoRefreshFailures = 0;
      this._autoRefreshTimer = setInterval(async () => {
        try {
          await this.refreshCurrentView();
          this._autoRefreshFailures = 0;
        } catch(e) {
          this._autoRefreshFailures++;
          if (this._autoRefreshFailures >= 3) {
            this._stopAutoRefresh();
            this._showError('Auto-refresh disabled after 3 failures');
          }
        }
      }, AUTO_REFRESH_INTERVAL_MS);
    },

    _stopAutoRefresh() {
      if (this._autoRefreshTimer) {
        clearInterval(this._autoRefreshTimer);
        this._autoRefreshTimer = null;
      }
    },

    async exportPredictions() {
      try {
        const data = await this._fetchWithRetry('/api/export/predictions');
        const csv = data.csv || '';
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `predictions-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch(e) {
        this._showError('Failed to export predictions');
        console.error(e);
      }
    },

    addComparisonMatch(matchId) {
      if (!this.comparisonMatches.includes(matchId)) {
        this.comparisonMatches.push(matchId);
      }
      if (this.comparisonMatches.length >= 2) {
        this.showComparisonModal = true;
      }
    },

    removeComparisonMatch(matchId) {
      this.comparisonMatches = this.comparisonMatches.filter(id => id !== matchId);
    },

    clearComparison() {
      this.comparisonMatches = [];
      this.showComparisonModal = false;
    },

    // ═══════ COMPUTED ═══════
    get leagues() {
      const s = new Set(this.matches.map(m => m.competition));
      return ['all', ...Array.from(s).sort()];
    },

    get filteredMatches() {
      let filtered = this.league === 'all'
        ? [...this.matches]
        : this.matches.filter(m => m.competition === this.league);

      // Team search filter - use debounced search query
      if (this._debouncedSearchQuery && this._debouncedSearchQuery.trim()) {
        const q = this._debouncedSearchQuery.trim().toLowerCase();
        filtered = filtered.filter(m =>
          (m.home_team || '').toLowerCase().includes(q) ||
          (m.away_team || '').toLowerCase().includes(q)
        );
      }

      if (this.sortMode === 'confidence') {
        filtered.sort((a, b) => {
          const aMax = Math.max(a.p_home ?? 0, a.p_draw ?? 0, a.p_away ?? 0);
          const bMax = Math.max(b.p_home ?? 0, b.p_draw ?? 0, b.p_away ?? 0);
          return bMax - aMax;
        });
      } else if (this.sortMode === 'edge') {
        filtered.sort((a, b) => {
          const aMax = Math.max(a.btts ?? 0, a.o25 ?? 0, a.p_home ?? 0, a.p_draw ?? 0, a.p_away ?? 0);
          const bMax = Math.max(b.btts ?? 0, b.o25 ?? 0, b.p_home ?? 0, b.p_draw ?? 0, b.p_away ?? 0);
          return bMax - aMax;
        });
      } else {
        filtered.sort((a, b) => String(a.utc_date).localeCompare(String(b.utc_date)));
      }
      return filtered;
    },

    get matchSummary() {
      const items = this.filteredMatches;
      const withPreds = items.filter(m => m.p_home != null);
      const strongest = withPreds.reduce((best, m) => {
        const bestP = best ? Math.max(best.p_home ?? 0, best.p_draw ?? 0, best.p_away ?? 0) : -1;
        const currP = Math.max(m.p_home ?? 0, m.p_draw ?? 0, m.p_away ?? 0);
        return currP > bestP ? m : best;
      }, null);
      return {
        total: items.length,
        predicted: withPreds.length,
        avgConfidence: withPreds.length > 0
          ? Math.round((withPreds.reduce((s, m) => s + Math.max(m.p_home ?? 0, m.p_draw ?? 0, m.p_away ?? 0), 0) / withPreds.length) * 100)
          : 0,
        modelAgreement: 'Pending', // From API
        strongest,
      };
    },

    groupedMatches() {
      const groups = {};
      for (const m of this.filteredMatches) {
        const d = m.utc_date.slice(0, 10);
        if (!groups[d]) groups[d] = [];
        groups[d].push(m);
      }
      return Object.entries(groups);
    },

    // ═══════ FORMATTERS ═══════
    pct(v) { return v != null ? Math.round(v * 100) + '%' : '—'; },
    pct1(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—'; },

    formatDate(s) {
      const d = new Date(s + 'T00:00:00');
      const now = new Date();
      const tomorrow = new Date(now); tomorrow.setDate(tomorrow.getDate() + 1);
      const localToday = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
      const localTmrw = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth()+1).padStart(2,'0')}-${String(tomorrow.getDate()).padStart(2,'0')}`;
      if (s === localToday) return 'Today';
      if (s === localTmrw) return 'Tomorrow';
      return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
    },

    formatRelativeDate(isoString) {
      const date = new Date(isoString);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
    },

    matchTime(m) {
      if (!m?.utc_date) return '';
      if (m.utc_date.length <= 10) return '';
      try {
        const d = new Date(m.utc_date.endsWith('Z') ? m.utc_date : m.utc_date + 'Z');
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } catch (_) {
        return m.utc_date.slice(11, 16);
      }
    },

    badgeClass(comp) {
      const map = {
        PL: 'badge-PL', SA: 'badge-SA', PD: 'badge-PD', BL1: 'badge-BL1', FL1: 'badge-FL1',
        DED: 'badge-DED', PPL: 'badge-PPL', ELC: 'badge-ELC', TR1: 'badge-TR1', BEL: 'badge-BEL',
        SL: 'badge-SL', A1: 'badge-A1', GR1: 'badge-GR1', SWS: 'badge-SWS', DK1: 'badge-DK1',
        SE1: 'badge-SE1', NO1: 'badge-NO1', PL1: 'badge-PL1'
      };
      return 'badge ' + (map[comp] || 'badge-default');
    },

    confidence(m) {
      if (m.p_home == null) return null;
      const mx = Math.max(m.p_home, m.p_draw, m.p_away);
      if (mx >= 0.55) return 'high';
      if (mx >= 0.42) return 'medium';
      return 'low';
    },

    confLabel(m) {
      const c = this.confidence(m);
      return c === 'high' ? 'Strong' : c === 'medium' ? 'Moderate' : c === 'low' ? 'Close' : '';
    },

    verdict(m) {
      if (m.p_home == null) return { text: '', cls: '' };
      const h = m.p_home, d = m.p_draw, a = m.p_away;
      const mx = Math.max(h, d, a);
      if (mx === h) {
        if (h >= 0.55) return { text: `${m.home_team} expected to win`, cls: 'verdict-home' };
        if (h >= 0.42) return { text: `${m.home_team} slight edge`, cls: 'verdict-home' };
        return { text: 'Tight match, home lean', cls: 'verdict-home' };
      }
      if (mx === a) {
        if (a >= 0.55) return { text: `${m.away_team} expected to win`, cls: 'verdict-away' };
        if (a >= 0.42) return { text: `${m.away_team} slight edge`, cls: 'verdict-away' };
        return { text: 'Tight match, away lean', cls: 'verdict-away' };
      }
      return { text: 'Draw is the highest probability', cls: 'verdict-draw' };
    },

    isHighest(outcome) {
      const p = this.md?.prediction;
      if (!p) return false;
      const mx = Math.max(p.p_home, p.p_draw, p.p_away);
      if (outcome === 'home') return p.p_home === mx;
      if (outcome === 'draw') return p.p_draw === mx;
      return p.p_away === mx;
    },

    winner(hg, ag) {
      return hg > ag ? 'home' : ag > hg ? 'away' : 'draw';
    },

    kelly(prob, odds) {
      if (!odds || odds <= 1) return 0;
      return Math.max(0, (prob * odds - 1) / (odds - 1));
    },

    kellyPct(prob, odds) {
      const k = this.kelly(prob, odds);
      return k > 0 ? (k * 100).toFixed(1) + '%' : '—';
    },

    consensusScore(experts) {
      if (!experts?.experts) return 50;
      const vals = Object.values(experts.experts);
      if (vals.length < 2) return 100;
      const outcomes = ['home', 'draw', 'away'];
      let totalWeight = 0, weightedVariance = 0;
      for (const outcome of outcomes) {
        const probs = vals.map(e => e.probs[outcome] || 0);
        const confs = vals.map(e => e.confidence || 0.5);
        const wSum = confs.reduce((a, b) => a + b, 0);
        const wMean = probs.reduce((a, p, i) => a + p * confs[i], 0) / (wSum || 1);
        const wVar = probs.reduce((a, p, i) => a + confs[i] * (p - wMean) ** 2, 0) / (wSum || 1);
        weightedVariance += wVar;
        totalWeight += 1;
      }
      const avgVar = weightedVariance / (totalWeight || 1);
      return Math.round((1 - Math.min(avgVar / 0.04, 1)) * 100);
    },

    consensusLabel(experts) {
      const s = this.consensusScore(experts);
      return s >= 75 ? 'Strong agreement' : s >= 45 ? 'Mixed signals' : 'Expert clash';
    },

    consensusClass(experts) {
      const s = this.consensusScore(experts);
      return s >= 75 ? 'agree' : s >= 45 ? 'mixed' : 'clash';
    },

    edge(modelProb, odds) {
      if (!odds || odds <= 1 || modelProb == null) return null;
      return modelProb - (1 / odds);
    },

    edgeText(modelProb, odds) {
      const e = this.edge(modelProb, odds);
      if (e == null) return '';
      const p = (e * 100).toFixed(1);
      return e > 0 ? `+${p}%` : `${p}%`;
    },

    leagueName(code) {
      const map = {
        PL: 'Premier League', SA: 'Serie A', PD: 'La Liga',
        BL1: 'Bundesliga', FL1: 'Ligue 1', ELC: 'Championship',
        DED: 'Eredivisie', PPL: 'Primeira Liga', TR1: 'Süper Lig',
        BEL: 'Pro League', SL: 'Premiership', A1: 'Bundesliga (AT)',
        GR1: 'Super League', SWS: 'Super League (CH)',
        DK1: 'Superliga', SE1: 'Allsvenskan', NO1: 'Eliteserien',
        PL1: 'Ekstraklasa'
      };
      return map[code] || code;
    },

    formatUpdated(ts) {
      if (!ts) return 'Never';
      const d = new Date(ts.replace(' ', 'T'));
      if (isNaN(d.getTime())) return ts;
      const diff = Math.floor((new Date() - d) / 60000);
      if (diff < 60) return `${diff}m ago`;
      if (diff < 1440) return `${Math.floor(diff/60)}h ago`;
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
    },

    perfAccuracyColor(acc) {
      if (acc >= 0.55) return 'var(--green)';
      if (acc >= 0.45) return 'var(--amber)';
      return 'var(--red)';
    },

    loglossGrade(ll) {
      if (ll == null) return '—';
      if (ll < 0.95) return 'Excellent';
      if (ll < 1.05) return 'Good';
      if (ll < 1.10) return 'Fair';
      return 'Needs work';
    },

    driftStatus(drift) {
      if (!drift) return { text: 'Unknown', cls: 'drift-unknown' };
      if (drift.drifted) return { text: 'Drift Detected', cls: 'drift-alert' };
      return { text: 'Stable', cls: 'drift-ok' };
    },

    destroy() {
      this._stopAutoRefresh();
    },
  }));

});

// Register Service Worker for PWA
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}
