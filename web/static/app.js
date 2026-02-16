/**
 * Footy Predictor v5 — Consolidated UI
 * Migrated all Streamlit features into FastAPI + Alpine.js SPA
 * URL-driven routing · Auto-loading · Performance dashboard
 */
document.addEventListener('alpine:init', () => {

  Alpine.data('app', () => ({
    // ── routing ──
    view: 'main',       // 'main' or 'match'
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
    model: 'v10_council',

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

    // ── match detail state ──
    md: null,           // match detail data
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

    // ── helpers ──
    async _fetch(url) {
      const r = await fetch(url);
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${r.status}`);
      }
      return r.json();
    },

    // ── lifecycle ──
    init() {
      // Detect URL for routing
      const path = window.location.pathname;
      const matchRoute = path.match(/^\/match\/(\d+)$/);
      if (matchRoute) {
        this.openMatch(parseInt(matchRoute[1]));
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
    },

    // ═══════ ROUTING ═══════
    goHome() {
      this.view = 'main';
      if (!this.matches.length) this.fetchMatches();
      history.pushState({ view: 'main' }, '', '/');
      document.title = 'Footy Predictor';
    },

    async openMatch(matchId, pushState = true) {
      this.view = 'match';
      this.matchId = matchId;
      this.md = null;
      this.matchExperts = null;
      this.matchH2H = null;
      this.matchForm = null;
      this.matchNarrative = null;
      this.matchXG = null;
      this.matchPatterns = null;
      this.loadingDetail = true;
      this.loadingExperts = false;
      this.loadingH2H = false;
      this.loadingAI = false;
      this.loadingXG = false;
      this.loadingPatterns = false;

      if (pushState) {
        history.pushState({ view: 'match', matchId }, '', `/match/${matchId}`);
      }
      window.scrollTo(0, 0);

      try {
        this.md = await this._fetch(`/api/matches/${matchId}?model=${this.model}`);
        document.title = `${this.md.home_team} vs ${this.md.away_team} — Footy Predictor`;
      } catch(e) { console.error(e); }
      this.loadingDetail = false;

      // Auto-load all match data in parallel
      this._loadMatchExperts(matchId);
      this._loadMatchH2H(matchId);
      this._loadMatchForm(matchId);
      this._loadMatchXG(matchId);
      this._loadMatchPatterns(matchId);
    },

    async _loadMatchExperts(matchId) {
      this.loadingExperts = true;
      try { this.matchExperts = await this._fetch(`/api/matches/${matchId}/experts`); } catch(e) { console.error(e); }
      this.loadingExperts = false;
    },

    async _loadMatchH2H(matchId) {
      this.loadingH2H = true;
      try { this.matchH2H = await this._fetch(`/api/matches/${matchId}/h2h`); } catch(e) { console.error(e); }
      this.loadingH2H = false;
    },

    async _loadMatchForm(matchId) {
      try { this.matchForm = await this._fetch(`/api/matches/${matchId}/form`); } catch(e) { console.error(e); }
    },

    async _loadMatchXG(matchId) {
      this.loadingXG = true;
      try { this.matchXG = await this._fetch(`/api/matches/${matchId}/xg`); } catch(e) { console.error(e); }
      this.loadingXG = false;
    },

    async _loadMatchPatterns(matchId) {
      this.loadingPatterns = true;
      try { this.matchPatterns = await this._fetch(`/api/matches/${matchId}/patterns`); } catch(e) { console.error(e); }
      this.loadingPatterns = false;
    },

    async loadAI() {
      if (!this.matchId || this.matchNarrative) return;
      this.loadingAI = true;
      try {
        const d = await this._fetch(`/api/matches/${this.matchId}/ai`);
        this.matchNarrative = d.narrative || 'AI analysis unavailable — is Ollama running?';
      } catch(e) { this.matchNarrative = 'Failed to generate analysis.'; }
      this.loadingAI = false;
    },

    // ═══════ API CALLS ═══════
    async fetchMatches() {
      this.loading = true;
      try {
        const d = await this._fetch(`/api/matches?days=${this.days}&model=${this.model}`);
        this.matches = d.matches || [];
      } catch(e) { console.error(e); }
      this.loading = false;
    },

    async fetchLastUpdated() {
      try {
        const d = await this._fetch('/api/last-updated');
        this.lastUpdated = d.last_updated;
      } catch(e) { /* ignore */ }
    },

    async fetchValueBets() {
      this.loadingBets = true;
      try {
        const d = await this._fetch('/api/insights/value-bets?min_edge=0.03');
        this.valueBets = d.bets || [];
      } catch(e) { console.error(e); }
      this.loadingBets = false;
    },

    async fetchBttsOu() {
      this.loadingBtts = true;
      try { this.bttsOu = await this._fetch('/api/insights/btts-ou'); } catch(e) { console.error(e); }
      this.loadingBtts = false;
    },

    async fetchAccumulators() {
      this.loadingAccas = true;
      try {
        const d = await this._fetch('/api/insights/accumulators');
        this.accumulators = d.accumulators || [];
      } catch(e) { console.error(e); }
      this.loadingAccas = false;
    },

    async fetchFormTable() {
      this.loadingFormTable = true;
      try {
        const d = await this._fetch(`/api/insights/form-table/${this.formTableComp}`);
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
      try { this.accuracyStats = await this._fetch(`/api/insights/accuracy?days_back=${this.accuracyDays}`); } catch(e) { console.error(e); }
      this.loadingAccuracy = false;
    },

    async fetchRoundPreview() {
      this.loadingRoundPreview = true;
      try { this.roundPreview = await this._fetch(`/api/insights/round-preview/${this.roundPreviewComp}`); } catch(e) { console.error(e); }
      this.loadingRoundPreview = false;
    },

    selectRoundPreviewComp(c) {
      this.roundPreviewComp = c;
      this.fetchRoundPreview();
    },

    async fetchPostMatchReview() {
      this.loadingReview = true;
      try { this.postMatchReview = await this._fetch('/api/insights/post-match-review?days_back=7'); } catch(e) { console.error(e); }
      this.loadingReview = false;
    },

    async fetchTrainingStatus() {
      this.loadingTraining = true;
      try { this.trainingStatus = await this._fetch('/api/training/status'); } catch(e) { console.error(e); }
      this.loadingTraining = false;
    },

    async fetchStats() {
      this.loadingStats = true;
      try { this.stats = await this._fetch('/api/stats'); } catch(e) { console.error(e); }
      this.loadingStats = false;
    },

    async fetchPerformance() {
      this.loadingPerf = true;
      try { this.performance = await this._fetch(`/api/performance?model=${this.model}`); } catch(e) { console.error(e); }
      this.loadingPerf = false;
    },

    async fetchLeagueTable() {
      this.loadingTable = true;
      try {
        const d = await this._fetch(`/api/league-table/${this.tableComp}`);
        this.leagueTable = d.standings || [];
      } catch(e) { console.error(e); this.leagueTable = []; }
      this.loadingTable = false;
    },

    selectTableComp(c) {
      this.tableComp = c;
      this.fetchLeagueTable();
    },

    switchTab(t) {
      this.tab = t;
      if (t === 'insights' && !this.valueBets.length) this.fetchValueBets();
      if (t === 'btts' && !this.bttsOu) this.fetchBttsOu();
      if (t === 'accas' && !this.accumulators.length) this.fetchAccumulators();
      if (t === 'form' && !this.formTable.length) this.fetchFormTable();
      if (t === 'accuracy' && !this.accuracyStats) this.fetchAccuracy();
      if (t === 'review' && !this.postMatchReview) { this.fetchPostMatchReview(); this.fetchRoundPreview(); }
      if (t === 'stats' && !this.stats) { this.fetchStats(); this.fetchPerformance(); }
      if (t === 'table' && !this.leagueTable.length) this.fetchLeagueTable();
      if (t === 'training' && !this.trainingStatus) this.fetchTrainingStatus();
    },

    // ═══════ COMPUTED ═══════
    get leagues() {
      const s = new Set(this.matches.map(m => m.competition));
      return ['all', ...Array.from(s).sort()];
    },

    get filteredMatches() {
      if (this.league === 'all') return this.matches;
      return this.matches.filter(m => m.competition === this.league);
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
      if (s === now.toISOString().slice(0,10)) return 'Today';
      if (s === tomorrow.toISOString().slice(0,10)) return 'Tomorrow';
      return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
    },

    matchTime(m) {
      if (!m?.utc_date) return '';
      return m.utc_date.length > 10 ? m.utc_date.slice(11, 16) : '';
    },

    badgeClass(comp) {
      const map = { PL: 'badge-PL', SA: 'badge-SA', PD: 'badge-PD', BL1: 'badge-BL1', FL1: 'badge-FL1' };
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
      const probs = vals.map(e => e.probs.home);
      const mean = probs.reduce((a,b) => a+b, 0) / probs.length;
      const variance = probs.reduce((a,b) => a + (b - mean) ** 2, 0) / probs.length;
      return Math.round((1 - Math.min(variance / 0.04, 1)) * 100);
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
      const map = { PL: 'Premier League', SA: 'Serie A', PD: 'La Liga', BL1: 'Bundesliga', FL1: 'Ligue 1', ELC: 'Championship' };
      return map[code] || code;
    },

    formatUpdated(ts) {
      if (!ts) return 'Never';
      const d = new Date(ts);
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
  }));
});
