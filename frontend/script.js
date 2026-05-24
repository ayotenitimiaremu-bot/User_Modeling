const API_BASE    = 'http://localhost:8000';
const RESULTS_KEY = 'smartbuyng_results';
const QUERY_KEY   = 'smartbuyng_query';

function formatNaira(amount) {
  if (!amount && amount !== 0) return '—';
  return '₦' + Number(amount).toLocaleString('en-NG');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.classList.toggle('btn--loading', loading);
}

function showError(msg) {
  const box = document.getElementById('errorBox');
  if (!box) return;
  box.textContent = msg;
  box.classList.add('visible');
}

function clearError() {
  const box = document.getElementById('errorBox');
  if (box) box.classList.remove('visible');
}

const PLATFORM_STYLES = {
  jumia: { label: 'Official', color: '#15803d', bg: '#f0fdf4', border: '#86efac' },
  konga: { label: 'Konga',    color: '#1d4ed8', bg: '#eff6ff', border: '#93c5fd' },
  jiji:  { label: 'Jiji',     color: '#b45309', bg: '#fffbeb', border: '#fcd34d' },
};

function getPlatformBadge(name) {
  const key = name.toLowerCase();
  const match = Object.keys(PLATFORM_STYLES).find(k => key.includes(k));
  if (!match) return '';
  const s = PLATFORM_STYLES[match];
  return `<span class="platform-badge" style="color:${s.color};background:${s.bg};border-color:${s.border}">${s.label}</span>`;
}

function buildTrustIndicator(score, type) {
  const fillClass = score >= 75 ? 'trust-fill--high' : score >= 45 ? 'trust-fill--mid' : 'trust-fill--low';
  const icon      = type === 'safe' ? '✓' : '⚠️';
  const iconClass = type === 'safe' ? 'trust-icon' : 'trust-icon';
  return `
    <div class="trust-indicator">
      <div class="trust-bar-full">
        <div class="trust-fill-bar ${fillClass}" style="width:${score}%"></div>
      </div>
      <div class="trust-meta">
        <span class="${iconClass}">${icon}</span>
        <span class="trust-pct">${score}% trusted</span>
      </div>
    </div>`;
}

function buildSafeCard(item) {
  const name  = item.name || item.title || item.platform || 'Product';
  const score = item.trust_score;
  const url   = item.url || '#';

  return `
    <div class="result-row">
      <span class="status-dot status-dot--green"></span>

      <div class="main-col">
        <div class="platform">${escHtml(name)} ${getPlatformBadge(name)}</div>
        ${item.seller ? `<div class="seller">${escHtml(item.seller)}</div>` : ''}
        <div class="detail-row">
          ${item.location ? `<span>📍 ${escHtml(item.location)}</span>` : ''}
          ${item.delivery ? `<span>🚚 ${escHtml(item.delivery)}</span>` : ''}
        </div>
        ${score !== undefined ? buildTrustIndicator(score, 'safe') : ''}
        <div class="card-actions">
          <a href="${escHtml(url)}" target="_blank" rel="noopener" class="btn--card-primary">View on ${escHtml(name)} →</a>
        </div>
      </div>

      <div class="price-col">
        <span class="price price--safe">${formatNaira(item.price)}</span>
        <span class="status-badge status-badge--green">Verified</span>
      </div>
    </div>`;
}

function buildScamCard(item) {
  const name = item.name || item.title || item.platform || 'Suspicious listing';
  const score = item.trust_score;

  const warningItems = Array.isArray(item.warnings)
    ? item.warnings
    : item.warning ? [item.warning] : [];

  const warningsHtml = warningItems.length ? `
    <div class="warning-box">
      <div class="warning-box-title">Why this was flagged:</div>
      ${warningItems.map(w => `
        <div class="warning-item">
          <span class="warn-icon">!</span>
          <span>${escHtml(w)}</span>
        </div>`).join('')}
    </div>` : '';

  return `
    <div class="result-row">
      <span class="status-dot status-dot--red"></span>

      <div class="main-col">
        <div class="platform">${escHtml(name)} ${getPlatformBadge(name)}</div>
        ${item.seller ? `<div class="seller">${escHtml(item.seller)}</div>` : ''}
        <div class="detail-row">
          ${item.location ? `<span>📍 ${escHtml(item.location)}</span>` : ''}
        </div>
        ${warningsHtml}
        ${score !== undefined ? buildTrustIndicator(score, 'scam') : ''}
      </div>

      <div class="price-col">
        ${item.price ? `<span class="price price--risk">${formatNaira(item.price)}</span>` : ''}
        <span class="status-badge status-badge--red">Flagged</span>
      </div>
    </div>`;
}

function handleSave(btn) {
  btn.textContent = '✓ Saved';
  btn.disabled = true;
  setTimeout(() => { btn.textContent = 'Save for later'; btn.disabled = false; }, 2000);
}

function handleReport(btn) {
  btn.textContent = '✓ Reported';
  btn.disabled = true;
}

function renderPriceComparison(data) {
  const el    = document.getElementById('priceComparison');
  if (!el) return;

  const safe  = data.safe_options  || [];
  const scams = data.scam_warnings || [];
  if (!safe.length || !scams.length) return;

  const cheapSafe = safe.reduce((m, i) => i.price < m.price ? i : m, safe[0]);
  const cheapScam = scams.reduce((m, i) => i.price < m.price ? i : m, scams[0]);
  const diff      = cheapSafe.price - cheapScam.price;
  const pct       = Math.round(Math.abs(diff / cheapSafe.price) * 100);
  const safeName  = cheapSafe.platform || cheapSafe.name || 'Safe listing';
  const scamName  = cheapScam.platform || cheapScam.name || 'Flagged listing';

  el.innerHTML = `
    <div class="price-comparison">
      <div class="pc-header">Price comparison</div>
      <div class="pc-row">
        <span class="pc-label">✅ Cheapest safe option</span>
        <span class="pc-price pc-price--safe">${formatNaira(cheapSafe.price)}</span>
        <span class="pc-source">${escHtml(safeName)}</span>
      </div>
      <div class="pc-row">
        <span class="pc-label">⚠️ Flagged cheap option</span>
        <span class="pc-price pc-price--risk">${formatNaira(cheapScam.price)}</span>
        <span class="pc-avoid">AVOID</span>
        <span class="pc-source">${escHtml(scamName)}</span>
      </div>
      <div class="pc-row">
        <span class="pc-label">Price difference</span>
        <span class="pc-diff">${formatNaira(Math.abs(diff))} — the flagged listing is ${pct}% cheaper but <strong>unsafe</strong></span>
      </div>
    </div>`;
}

function renderResults() {
  const raw      = sessionStorage.getItem(RESULTS_KEY);
  const queryRaw = sessionStorage.getItem(QUERY_KEY);

  if (!raw) { window.location.href = 'index.html'; return; }

  const data  = JSON.parse(raw);
  const query = queryRaw ? JSON.parse(queryRaw) : {};

  const title    = document.getElementById('resultsTitle');
  const subtitle = document.getElementById('resultsSubtitle');
  if (title)    title.textContent    = `"${query.query || 'Search'}"`;
  if (subtitle) subtitle.textContent = `${query.location || ''} · ${formatNaira(query.budget_min)} – ${formatNaira(query.budget_max)}`;

  if (data.savings && data.savings > 0) {
    const banner = document.getElementById('savingsBanner');
    const amount = document.getElementById('savingsAmount');
    if (banner) banner.style.display = 'flex';
    if (amount) amount.textContent   = formatNaira(data.savings);
  }

  renderPriceComparison(data);

  const safe  = data.safe_options  || [];
  const scams = data.scam_warnings || [];

  const safeList  = document.getElementById('safeList');
  const safeCount = document.getElementById('safeCount');
  if (safeCount) safeCount.textContent = safe.length;
  if (safeList) {
    safeList.innerHTML = safe.length
      ? safe.map(buildSafeCard).join('')
      : `<div class="empty-state">
           <div class="empty-icon">🔍</div>
           <div class="empty-title">No safe listings found</div>
           <div class="empty-body">Try adjusting your budget or broadening your search term</div>
         </div>`;
  }

  const scamList  = document.getElementById('scamList');
  const scamCount = document.getElementById('scamCount');
  if (scamCount) scamCount.textContent = scams.length;
  if (scamList) {
    scamList.innerHTML = scams.length
      ? scams.map(buildScamCard).join('')
      : `<div class="empty-state">
           <div class="empty-icon">✅</div>
           <div class="empty-title">No flagged listings detected</div>
           <div class="empty-body">All results look clean for this search</div>
         </div>`;
  }

  requestAnimationFrame(() => {
    const bar = document.querySelector('.results-bar');
    if (bar) bar.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  if (!safe.length && !scams.length) {
    const safeEl = document.getElementById('safeList');
    if (safeEl) safeEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📦</div>
        <div class="empty-title">No products found for ${escHtml(query.query || 'your search')}</div>
        <ul class="empty-suggestions">
          <li>Try a different search term</li>
          <li>Adjust your budget range</li>
          <li>Check spelling</li>
        </ul>
        <a href="index.html" class="btn btn--primary">Search again</a>
      </div>`;
  }
}

function runLoadingAnimation() {
  const steps = ['step-jumia', 'step-konga', 'step-jiji', 'step-scam'];
  const delays = [450, 900, 1350, 1750];

  return Promise.all(
    steps.map((id, i) =>
      new Promise(resolve => setTimeout(() => {
        const el = document.getElementById(id);
        if (el) el.classList.add('done');
        resolve();
      }, delays[i]))
    )
  ).then(() => new Promise(r => setTimeout(r, 300)));
}

const form = document.getElementById('searchForm');

if (form) {
  let rawMin = null;
  let rawMax = null;

  const validity   = { query: false, budget_min: false, budget_max: false, location: false };
  const submitBtn  = document.getElementById('submitBtn');
  const minInput   = document.getElementById('budget_min');
  const maxInput   = document.getElementById('budget_max');

  function setFieldError(fieldId, message) {
    const input = document.getElementById(fieldId);
    const err   = document.getElementById(`${fieldId}-error`);
    if (input) input.classList.toggle('input--error', !!message);
    if (err)   err.textContent = message || '';
  }

  function updateSubmitState() {
    submitBtn.disabled = !Object.values(validity).every(Boolean);
  }

  function applyThousands(input) {
    const digits = input.value.replace(/[^\d]/g, '');
    if (!digits) { input.value = ''; return null; }
    const num = parseInt(digits, 10);
    input.value = num.toLocaleString('en-US');
    return num;
  }

  function validateQuery() {
    const val = document.getElementById('query').value.trim();
    if (!val)           setFieldError('query', 'Please enter a product name.');
    else if (val.length < 3) setFieldError('query', 'Product name must be at least 3 characters.');
    else                setFieldError('query', '');
    validity.query = !!val && val.length >= 3;
    updateSubmitState();
  }

  function validateMin() {
    if (rawMin === null || rawMin === undefined) setFieldError('budget_min', 'Please enter a minimum budget.');
    else if (rawMin <= 0)                        setFieldError('budget_min', 'Minimum budget must be greater than 0.');
    else                                         setFieldError('budget_min', '');
    validity.budget_min = rawMin > 0;
    updateSubmitState();
  }

  function validateMax() {
    if (rawMax === null || rawMax === undefined)           setFieldError('budget_max', 'Please enter a maximum budget.');
    else if (rawMin !== null && rawMax <= rawMin) setFieldError('budget_max', 'Maximum must be greater than minimum.');
    else                                                   setFieldError('budget_max', '');
    validity.budget_max = rawMax > 0 && (rawMin === null || rawMax > rawMin);
    updateSubmitState();
  }

  function validateLocation() {
    const val = document.getElementById('location').value;
    setFieldError('location', val ? '' : 'Please select your location.');
    validity.location = !!val;
    updateSubmitState();
  }

  minInput.addEventListener('input', () => { rawMin = applyThousands(minInput); if (validity.budget_max === false) validateMax(); });
  maxInput.addEventListener('input', () => { rawMax = applyThousands(maxInput); });

  minInput.addEventListener('blur', () => { validateMin(); if (maxInput.dataset.touched) validateMax(); });
  maxInput.addEventListener('blur', () => { maxInput.dataset.touched = '1'; validateMax(); });

  document.getElementById('query').addEventListener('blur', validateQuery);
  document.getElementById('query').addEventListener('input', () => { if (!validity.query) validateQuery(); });
  document.getElementById('location').addEventListener('change', validateLocation);

  submitBtn.disabled = true;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();
    validateQuery(); validateMin(); validateMax(); validateLocation();
    if (!Object.values(validity).every(Boolean)) return;

    const query    = document.getElementById('query').value.trim();
    const location = document.getElementById('location').value;
    const priorities = [...document.querySelectorAll('input[name="priorities"]:checked')].map(cb => cb.value);
    const payload  = { query, budget_min: rawMin, budget_max: rawMax, location, priorities };

    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'flex';

    try {
      const [data] = await Promise.all([
        fetch(`${API_BASE}/analyze-product`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }).then(res => {
          if (!res.ok) return res.json().then(e => { throw new Error(e.detail || `Server error (${res.status})`); });
          return res.json();
        }),
        runLoadingAnimation(),
      ]);

      sessionStorage.setItem(RESULTS_KEY, JSON.stringify(data));
      sessionStorage.setItem(QUERY_KEY,   JSON.stringify({ query, budget_min: rawMin, budget_max: rawMax, location }));
      window.location.href = 'results.html';

    } catch (err) {
      if (overlay) overlay.style.display = 'none';
      showError(
        err.name === 'TypeError'
          ? 'Could not reach the server. Make sure the backend is running on localhost:8000.'
          : err.message || 'Something went wrong. Please try again.'
      );
      setLoading(submitBtn, false);
    }
  });
}

const shareBtn = document.getElementById('shareBtn');
if (shareBtn) {
  shareBtn.addEventListener('click', async () => {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(window.location.href);
      const orig = shareBtn.textContent;
      shareBtn.textContent = 'Copied!';
      setTimeout(() => { shareBtn.textContent = orig; }, 2000);
    }
  });
}

if (document.getElementById('safeList')) renderResults();
