/**
 * Footy Predictor v3 — Enhanced frontend
 * League filter · Confidence tiers · Verdicts · Kelly criterion
 * Auto-load · Performance tracking · Poisson stats · Scored predictions
 */
document.addEventListener('alpine:init', () => {

  Alpine.data('app', () => ({
    // ── state ──
    tab: 'matches',
    matches: [],
    loading: true,
    league: 'all',
    detailOpen: false,
    detailMatch: null,
    detailData: null,
    detailExperts: null,
    detailH2H: null,
    detailNarrative: null,
    detailForm: null,
    loadingDetail: false,
    loadingExperts: false,
    loadingH2H: false,
    loadingAI: false,
    valueBets: [],
    loadingBets: false,
    stats: null,
    loadingStats: false,
    performance: null,
    loadingPerf: false,
    lastUpdated: null,
    days: 14,
    model: 'v7_council',

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
      this.fetchMatches();
      this.fetchLastUpdated();
    },

    // ═══════ API ═══════
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

    async openDetail(match) {
      this.detailMatch = match;
      this.detailData = null;
      this.detailExperts = null;
      this.detailH2H = null;
      this.detailNarrative = null;
      this.detailForm = null;
      this.detailOpen = true;
      this.loadingDetail = true;
      this.loadingExperts = false;
      this.loadingH2H = false;
      this.loadingAI = false;

      try {
        this.detailData = await this._fetch(`/api/matches/${match.match_id}?model=${this.model}`);
      } catch(e) { console.error(e); }
      this.loadingDetail = false;

      // Auto-load everything in parallel (no manual button clicks)
      this.loadExperts();
      this.loadForm();
      this.loadH2H();
    },

    closeDetail() { this.detailOpen = false; },

    async loadExperts() {
      if (!this.detailMatch || this.detailExperts) return;
      this.loadingExperts = true;
      try {
        this.detailExperts = await this._fetch(`/api/matches/${this.detailMatch.match_id}/experts`);
      } catch(e) { console.error(e); }
      this.loadingExperts = false;
    },

    async loadH2H() {
      if (!this.detailMatch || this.detailH2H) return;
      this.loadingH2H = true;
      try {
        this.detailH2H = await this._fetch(`/api/matches/${this.detailMatch.match_id}/h2h`);
      } catch(e) { console.error(e); }
      this.loadingH2H = false;
    },

    async loadForm() {
      if (!this.detailMatch) return;
      try {
        this.detailForm = await this._fetch(`/api/matches/${this.detailMatch.match_id}/form`);
      } catch(e) { console.error(e); }
    },

    async loadAI() {
      if (!this.detailMatch || this.detailNarrative) return;
      this.loadingAI = true;
      try {
        const d = await this._fetch(`/api/matches/${this.detailMatch.match_id}/ai`);
        this.detailNarrative = d.narrative || 'AI analysis unavailable — is Ollama running?';
      } catch(e) {
        this.detailNarrative = 'Failed to generate analysis.';
      }
      this.loadingAI = false;
    },

    async fetchValueBets() {
      this.loadingBets = true;
      try {
        const d = await this._fetch('/api/insights/value-bets?min_edge=0.03');
        this.valueBets = d.bets || [];
      } catch(e) { console.error(e); }
      this.loadingBets = false;
    },

    async fetchStats() {
      this.loadingStats = true;
      try {
        this.stats = await this._fetch('/api/stats');
      } catch(e) { console.error(e); }
      this.loadingStats = false;
    },

    async fetchPerformance() {
      this.loadingPerf = true;
      try {
        this.performance = await this._fetch(`/api/performance?model=${this.model}`);
      } catch(e) { console.error(e); }
      this.loadingPerf = false;
    },

    switchTab(t) {
      this.tab = t;
      if (t === 'insights' && !this.valueBets.length) this.fetchValueBets();
      if (t === 'stats' && !this.stats) { this.fetchStats(); this.fetchPerformance(); }
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

    // ═══════ HELPERS ═══════
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
      return m.utc_date.length > 10 ? m.utc_date.slice(11, 16) : '';
    },

    badgeClass(comp) {
      const map = { PL: 'badge-PL', SA: 'badge-SA', PD: 'badge-PD', BL1: 'badge-BL1' };
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
      if (c === 'high') return 'Strong';
      if (c === 'medium') return 'Moderate';
      if (c === 'low') return 'Close';
      return '';
    },

    verdict(m) {
      if (m.p_home == null) return { text: '', cls: '' };
      const h = m.p_home, d = m.p_draw, a = m.p_away;
      const mx = Math.max(h, d, a);
      if (mx === h) {
        if (h >= 0.55) return { text: `${m.home_team} expected to win`, cls: 'verdict-home' };
        if (h >= 0.42) return { text: `${m.home_team} slight edge`, cls: 'verdict-home' };
        return { text: 'Tight match, home slight lean', cls: 'verdict-home' };
      }
      if (mx === a) {
        if (a >= 0.55) return { text: `${m.away_team} expected to win`, cls: 'verdict-away' };
        if (a >= 0.42) return { text: `${m.away_team} slight edge`, cls: 'verdict-away' };
        return { text: 'Tight match, away slight lean', cls: 'verdict-away' };
      }
      return { text: 'Draw is the highest probability', cls: 'verdict-draw' };
    },

    isHighest(outcome, m) {
      if (!m) return false;
      const data = this.detailData?.prediction || m;
      const h = data.p_home, d = data.p_draw, a = data.p_away;
      if (h == null) return false;
      const mx = Math.max(h, d, a);
      if (outcome === 'home') return h === mx;
      if (outcome === 'draw') return d === mx;
      return a === mx;
    },

    winner(hg, ag) {
      if (hg > ag) return 'home';
      if (ag > hg) return 'away';
      return 'draw';
    },

    kelly(prob, odds) {
      if (!odds || odds <= 1) return 0;
      const k = (prob * odds - 1) / (odds - 1);
      return Math.max(0, k);
    },

    kellyPct(prob, odds) {
      const k = this.kelly(prob, odds);
      return k > 0 ? (k * 100).toFixed(1) + '%' : '—';
    },

    consensusScore() {
      if (!this.detailExperts?.experts) return 50;
      const experts = Object.values(this.detailExperts.experts);
      if (experts.length < 2) return 100;
      const homeProbs = experts.map(e => e.probs.home);
      const mean = homeProbs.reduce((a,b) => a+b, 0) / homeProbs.length;
      const variance = homeProbs.reduce((a,b) => a + (b - mean) ** 2, 0) / homeProbs.length;
      const normalised = 1 - Math.min(variance / 0.04, 1);
      return Math.round(normalised * 100);
    },

    consensusLabel() {
      const s = this.consensusScore();
      if (s >= 75) return 'Strong agreement';
      if (s >= 45) return 'Mixed signals';
      return 'Expert clash';
    },

    consensusClass() {
      const s = this.consensusScore();
      if (s >= 75) return 'agree';
      if (s >= 45) return 'mixed';
      return 'clash';
    },

    edge(modelProb, odds) {
      if (!odds || odds <= 1 || modelProb == null) return null;
      return modelProb - (1 / odds);
    },

    edgeText(modelProb, odds) {
      const e = this.edge(modelProb, odds);
      if (e == null) return '';
      const pct = (e * 100).toFixed(1);
      return e > 0 ? `+${pct}%` : `${pct}%`;
    },

    leagueName(code) {
      const map = { PL: 'Premier League', SA: 'Serie A', PD: 'La Liga', BL1: 'Bundesliga', FL1: 'Ligue 1', ELC: 'Championship' };
      return map[code] || code;
    },

    formatUpdated(ts) {
      if (!ts) return 'Never';
      const d = new Date(ts);
      const now = new Date();
      const diff = Math.floor((now - d) / 60000);
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
  }));
});
