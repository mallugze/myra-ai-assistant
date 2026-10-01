/**
 * MYRA NEURAL MONITOR - CLIENT LOGIC & REAL-TIME TELEMETRY
 * Polls backend endpoints, renders dynamic skill diagnostic cards,
 * pinpoints root causes, and manages interactive tests.
 */

// State
let autoRefresh = true;
let pollIntervalMs = 1500;
let pollTimer = null;
let currentFilter = 'ALL';
let autoScroll = true;
let isCamActive = true;
let testingSkills = new Set();

// DOM Selectors
const masterStatusCard = document.getElementById('master-status-card');
const masterBeacon = document.getElementById('master-beacon');
const masterStatusLabel = document.getElementById('master-status-label');
const masterStatusSubtext = document.getElementById('master-status-subtext');
const globalPulse = document.getElementById('global-pulse');

const uptimeVal = document.getElementById('uptime-val');
const targetVal = document.getElementById('target-val');
const lastUpdatedText = document.getElementById('last-updated-text');

const alertBannerContainer = document.getElementById('alert-banner-container');
const alertBannerTitle = document.getElementById('alert-banner-title');
const alertBannerDesc = document.getElementById('alert-banner-desc');
const btnJumpErrors = document.getElementById('btn-jump-errors');

const chipActiveCount = document.getElementById('chip-active-count');
const chipBrainLatency = document.getElementById('chip-brain-latency');
const chipVisionStatus = document.getElementById('chip-vision-status');
const chipAudioRoute = document.getElementById('chip-audio-route');
const chipMemoryTurns = document.getElementById('chip-memory-turns');

const skillsGrid = document.getElementById('skills-grid');
const incidentCounts = document.getElementById('incident-counts');
const incidentList = document.getElementById('incident-list');
const consoleTerminal = document.getElementById('console-terminal');

const cameraFeedImg = document.getElementById('camera-feed-img');
const btnCamPower = document.getElementById('btn-cam-power');
const camPowerIcon = document.getElementById('cam-power-icon');
const camPowerText = document.getElementById('cam-power-text');
const visionFeedBadge = document.getElementById('vision-feed-badge');
const statTargetName = document.getElementById('stat-target-name');
const statPeopleCount = document.getElementById('stat-people-count');
const statObjects = document.getElementById('stat-objects');

const pptDeckFilename = document.getElementById('ppt-deck-filename');
const pptProgressBar = document.getElementById('ppt-progress-bar');
const pptSlideCounter = document.getElementById('ppt-slide-counter');
const pptSlideTitle = document.getElementById('ppt-slide-title');
const pptStatusBadge = document.getElementById('ppt-status-badge');

const currentExpression = document.getElementById('current-expression');
const lipSyncStatus = document.getElementById('lip-sync-status');

// Icons map for subsystems
const SUBSYSTEM_ICONS = {
  'BRAIN-01': '🧠',
  'EYES-02': '👁️',
  'VOICE-03': '🗣️',
  'EARS-04': '👂',
  'MEM-05': '💾',
  'AVTR-06': '🎭',
  'PPT-07': '📑',
  'ROUT-08': '🎧'
};

// ==========================================================================
// TELEMETRY FETCH & SYNC
// ==========================================================================

async function fetchTelemetry() {
  try {
    const res = await fetch('/api/monitor/skills');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderDashboard(data);
  } catch (err) {
    handleOfflineState(err);
  }
}

async function fetchLogs() {
  try {
    const res = await fetch(`/api/monitor/logs?limit=80&subsystem=${currentFilter === 'ALL' ? '' : currentFilter}`);
    if (!res.ok) return;
    const logs = await res.json();
    renderLogs(logs);
  } catch (err) {
    console.warn("Log fetch failed:", err);
  }
}

// ==========================================================================
// RENDER DASHBOARD
// ==========================================================================

function renderDashboard(data) {
  const overall = data.overall_status || 'OPERATIONAL';
  const subsystems = data.subsystems || [];
  const errors = data.active_errors || [];

  // 1. Master Status
  masterStatusCard.className = 'master-status-card';
  if (overall === 'CRITICAL_FAULT') {
    masterStatusCard.classList.add('status-error');
    masterStatusLabel.textContent = 'CRITICAL SYSTEM FAULT';
    masterStatusSubtext.textContent = `${errors.length} incident(s) requiring attention`;
    globalPulse.style.background = 'var(--neon-ruby)';
    globalPulse.style.boxShadow = '0 0 8px var(--neon-ruby)';
  } else if (overall === 'DEGRADED') {
    masterStatusCard.classList.add('status-warning');
    masterStatusLabel.textContent = 'PARTIAL WARNING';
    masterStatusSubtext.textContent = 'Some circuits running with degraded fallback';
    globalPulse.style.background = 'var(--neon-amber)';
    globalPulse.style.boxShadow = '0 0 8px var(--neon-amber)';
  } else {
    masterStatusLabel.textContent = 'ALL SYSTEMS NORMAL';
    masterStatusSubtext.textContent = '8 of 8 neural circuits active & healthy';
    globalPulse.style.background = 'var(--neon-emerald)';
    globalPulse.style.boxShadow = '0 0 8px var(--neon-emerald)';
  }

  // 2. Metrics & Timestamps
  uptimeVal.textContent = data.uptime || '--:--:--';
  lastUpdatedText.textContent = `Updated ${new Date().toLocaleTimeString()}`;

  // 3. Incident Alert Banner
  if (errors.length > 0) {
    alertBannerContainer.style.display = 'block';
    const criticalCount = errors.filter(e => e.severity === 'CRITICAL').length;
    alertBannerTitle.textContent = criticalCount > 0 
      ? `${criticalCount} Critical Subsystem Fault Detected` 
      : `${errors.length} System Warning(s) Detected`;
    alertBannerDesc.textContent = errors[0].error || 'Inspect root causes below for recommended fixes.';
  } else {
    alertBannerContainer.style.display = 'none';
  }

  // 4. Hero Summary Strip
  chipActiveCount.textContent = data.active_subsystems_count || '8 / 8 Active';
  
  const brainSub = subsystems.find(s => s.code === 'BRAIN-01');
  if (brainSub) {
    chipBrainLatency.textContent = brainSub.latency_ms ? `${brainSub.latency_ms} ms` : (brainSub.status === 'ONLINE' ? '< 150 ms' : 'Offline');
  }

  const eyesSub = subsystems.find(s => s.code === 'EYES-02');
  if (eyesSub) {
    setCameraUIState(eyesSub.camera_active);
    chipVisionStatus.textContent = eyesSub.camera_active ? 'Online (30 FPS)' : 'Standby (Hardware Released)';
    targetVal.textContent = eyesSub.tracked_target || 'Mallu';
    statTargetName.textContent = `Target: ${eyesSub.tracked_target || 'Mallu'}`;
    statPeopleCount.textContent = `People: ${eyesSub.person_count !== undefined ? eyesSub.person_count : 0}`;
    statObjects.textContent = `Objects: ${(eyesSub.detected_objects && eyesSub.detected_objects.length) ? eyesSub.detected_objects.slice(0, 2).join(', ') : 'None'}`;
  }

  const routerSub = subsystems.find(s => s.code === 'ROUT-08');
  if (routerSub) {
    chipAudioRoute.textContent = routerSub.is_headphones ? 'Rockerz 650 Pro' : 'Laptop Speakers';
  }

  const memSub = subsystems.find(s => s.code === 'MEM-05');
  if (memSub) {
    chipMemoryTurns.textContent = `${memSub.conversations_logged || 0} Turns`;
  }

  // 5. Render 8 Skill Cards
  renderSkillCards(subsystems);

  // 6. Render Incident Inspector
  renderIncidentInspector(errors);

  // 7. Update Presentation Panel
  const pptSub = subsystems.find(s => s.code === 'PPT-07');
  if (pptSub) {
    pptDeckFilename.textContent = pptSub.deck_file || 'myra.pptx';
    const cur = pptSub.current_slide || 1;
    const tot = pptSub.total_slides || 7;
    const pct = Math.min(100, Math.round((cur / tot) * 100));
    pptProgressBar.style.width = `${pct}%`;
    pptSlideCounter.textContent = `Slide ${cur} of ${tot}`;
    pptSlideTitle.textContent = pptSub.current_slide_title || `Slide ${cur}: Presentation in Progress`;
    pptStatusBadge.textContent = pptSub.is_presenting ? 'PRESENTING' : (pptSub.deck_exists ? 'READY' : 'MISSING');
    pptStatusBadge.className = `badge-tag ${pptSub.is_presenting ? 'badge-tag' : 'badge-emerald'}`;
  }
}

// ==========================================================================
// RENDER SKILL CARDS
// ==========================================================================

function renderSkillCards(subsystems) {
  skillsGrid.innerHTML = '';

  subsystems.forEach(sub => {
    const card = document.createElement('div');
    const statusLower = (sub.status || 'online').toLowerCase();
    card.className = `skill-card status-${statusLower}`;

    const icon = SUBSYSTEM_ICONS[sub.code] || '⚙️';
    const isTesting = testingSkills.has(sub.code);

    // Dynamic metrics grid
    let metricsHtml = '';
    if (sub.code === 'BRAIN-01') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Model</span><span class="m-val">${sub.active_model || 'myra:latest'}</span></div>
        <div class="metric-item"><span class="m-key">Port</span><span class="m-val">${sub.port || 11434}</span></div>
        <div class="metric-item"><span class="m-key">Latency</span><span class="m-val">${sub.latency_ms ? sub.latency_ms + 'ms' : 'N/A'}</span></div>
        <div class="metric-item"><span class="m-key">Repository</span><span class="m-val">${(sub.models_available && sub.models_available.length) ? sub.models_available.length + ' models' : 'Ollama'}</span></div>
      `;
    } else if (sub.code === 'EYES-02') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">YOLO Model</span><span class="m-val">${sub.yolo_weights || 'yolo11s.pt'}</span></div>
        <div class="metric-item"><span class="m-key">Face Engine</span><span class="m-val">InsightFace</span></div>
        <div class="metric-item"><span class="m-key">Target</span><span class="m-val">${sub.tracked_target || 'Mallu'}</span></div>
        <div class="metric-item"><span class="m-key">People In View</span><span class="m-val">${sub.person_count !== undefined ? sub.person_count : 0}</span></div>
      `;
    } else if (sub.code === 'VOICE-03') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Engine</span><span class="m-val">Silero v3_en</span></div>
        <div class="metric-item"><span class="m-key">Rate</span><span class="m-val">${sub.sample_rate || '48000 Hz'}</span></div>
        <div class="metric-item"><span class="m-key">Acronyms</span><span class="m-val">AI/LLM Expand</span></div>
        <div class="metric-item"><span class="m-key">Breath Pause</span><span class="m-val">0.55s Full Stop</span></div>
      `;
    } else if (sub.code === 'EARS-04') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Model</span><span class="m-val">${sub.engine || 'Faster-Whisper'}</span></div>
        <div class="metric-item"><span class="m-key">Hardware</span><span class="m-val">${sub.acceleration || 'CUDA'}</span></div>
        <div class="metric-item"><span class="m-key">Threshold</span><span class="m-val">${sub.rms_threshold || '0.0040'}</span></div>
        <div class="metric-item"><span class="m-key">Input Mic</span><span class="m-val" title="${sub.active_microphone}">${sub.is_headset_mic ? 'Headset Mic' : 'Laptop Mic'}</span></div>
      `;
    } else if (sub.code === 'MEM-05') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Database</span><span class="m-val">myra_memory.db</span></div>
        <div class="metric-item"><span class="m-key">Integrity</span><span class="m-val">${sub.integrity || 'ok'}</span></div>
        <div class="metric-item"><span class="m-key">Turns</span><span class="m-val">${sub.conversations_logged || 0}</span></div>
        <div class="metric-item"><span class="m-key">Faces</span><span class="m-val">${sub.known_faces_count || 1} Enrolled</span></div>
      `;
    } else if (sub.code === 'AVTR-06') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Port</span><span class="m-val">${sub.vmc_port || 39539}</span></div>
        <div class="metric-item"><span class="m-key">Protocol</span><span class="m-val">VMC OSC</span></div>
        <div class="metric-item"><span class="m-key">Lip-Sync</span><span class="m-val">VB-Audio Cable</span></div>
        <div class="metric-item"><span class="m-key">Expressions</span><span class="m-val">6 Blendshapes</span></div>
      `;
    } else if (sub.code === 'PPT-07') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Deck</span><span class="m-val">${sub.deck_file || 'myra.pptx'}</span></div>
        <div class="metric-item"><span class="m-key">Slide</span><span class="m-val">${sub.current_slide || 1} / ${sub.total_slides || 7}</span></div>
        <div class="metric-item"><span class="m-key">Automation</span><span class="m-val">Auto-Advance</span></div>
        <div class="metric-item"><span class="m-key">Mode</span><span class="m-val">7-Slide Script</span></div>
      `;
    } else if (sub.code === 'ROUT-08') {
      metricsHtml = `
        <div class="metric-item"><span class="m-key">Output</span><span class="m-val" title="${sub.output_device}">${sub.is_headphones ? 'Rockerz 650' : 'Laptop Speaker'}</span></div>
        <div class="metric-item"><span class="m-key">Input</span><span class="m-val" title="${sub.input_device}">${sub.is_headset_mic ? 'Headset Mic' : 'Laptop Mic'}</span></div>
        <div class="metric-item"><span class="m-key">Lip-Sync Bus</span><span class="m-val">${sub.vb_cable_lip_sync || 'Connected'}</span></div>
        <div class="metric-item"><span class="m-key">Tail Buffer</span><span class="m-val">400ms Safety</span></div>
      `;
    }

    // Inline error callout if present
    let errorCalloutHtml = '';
    if (sub.error) {
      errorCalloutHtml = `
        <div class="card-error-callout">
          <div class="cec-title">⚠ Incident Detected</div>
          <div class="cec-msg">${sub.error}</div>
          ${sub.fix_tip ? `<div class="cec-tip"><strong>Fix:</strong> ${sub.fix_tip}</div>` : ''}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="card-header-row">
        <div class="card-title-group">
          <div class="card-icon">${icon}</div>
          <div class="card-name-wrap">
            <span class="card-name">${sub.name}</span>
            <span class="card-code">${sub.code}</span>
          </div>
        </div>
        <span class="card-badge badge-${statusLower}">${sub.status}</span>
      </div>

      <div class="card-body-text">${sub.details || 'Subsystem online.'}</div>

      <div class="card-metrics-grid">
        ${metricsHtml}
      </div>

      ${errorCalloutHtml}

      <div class="card-footer-row">
        <span class="card-latency-tag">${sub.latency_ms ? `⚡ ${sub.latency_ms}ms` : 'Circuit Ready'}</span>
        <button class="btn-test-skill" data-code="${sub.code}" ${isTesting ? 'disabled' : ''}>
          ${isTesting ? '⏳ Testing...' : '⚡ Test Skill'}
        </button>
      </div>
    `;

    skillsGrid.appendChild(card);
  });

  // Attach event listeners to test buttons
  document.querySelectorAll('.btn-test-skill').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const code = e.currentTarget.getAttribute('data-code');
      triggerSkillTest(code);
    });
  });
}

// ==========================================================================
// RENDER INCIDENT INSPECTOR
// ==========================================================================

function renderIncidentInspector(errors) {
  if (!errors || errors.length === 0) {
    incidentCounts.innerHTML = '<span class="count-tag-healthy">✔ No Active Faults Detected</span>';
    incidentList.innerHTML = `
      <div class="incident-card-empty">
        <div class="empty-icon">✨</div>
        <div class="empty-title">All Subsystems Running Smoothly</div>
        <div class="empty-sub">No crashes, timeouts, or hardware disconnects detected across any of Myra's 8 circuits.</div>
      </div>
    `;
    return;
  }

  incidentCounts.innerHTML = `<span class="count-tag-fault">${errors.length} Active Incident(s)</span>`;
  incidentList.innerHTML = '';

  errors.forEach(err => {
    const isCrit = err.severity === 'CRITICAL';
    const incCard = document.createElement('div');
    incCard.className = `incident-card ${isCrit ? '' : 'incident-warn'}`;

    incCard.innerHTML = `
      <div class="inc-header">
        <span class="inc-subsystem-pill">${err.subsystem} [${err.code || 'SYS'}]</span>
        <span class="inc-severity-tag">${err.severity || 'WARNING'}</span>
      </div>
      <div class="inc-message">${err.error}</div>
      ${err.fix_tip ? `
        <div class="inc-fix-box">
          <div class="fix-icon">💡</div>
          <div class="fix-text">
            <strong>Recommended Solution:</strong> ${err.fix_tip}
          </div>
        </div>
      ` : ''}
    `;

    incidentList.appendChild(incCard);
  });
}

// ==========================================================================
// RENDER CONSOLE LOGS
// ==========================================================================

function renderLogs(logs) {
  if (!logs || logs.length === 0) return;

  consoleTerminal.innerHTML = '';
  logs.forEach(log => {
    const line = document.createElement('div');
    const levelLower = (log.level || 'info').toLowerCase();
    line.className = `log-line log-${levelLower}`;

    const subBadgeClass = getSubsystemBadgeClass(log.subsystem);

    line.innerHTML = `
      <span class="log-ts">${log.timestamp || '00:00:00'}</span>
      <span class="log-badge ${subBadgeClass}">${log.subsystem || 'SYS'}</span>
      <span class="log-msg">${escapeHtml(log.message || '')}</span>
    `;

    consoleTerminal.appendChild(line);
  });

  if (autoScroll) {
    consoleTerminal.scrollTop = consoleTerminal.scrollHeight;
  }
}

function getSubsystemBadgeClass(sub) {
  const s = (sub || '').toUpperCase();
  if (s.includes('BRAIN')) return 'badge-brain';
  if (s.includes('EYE') || s.includes('VISION')) return 'badge-eyes';
  if (s.includes('VOICE') || s.includes('TTS')) return 'badge-voice';
  if (s.includes('AUDIO') || s.includes('ROUT')) return 'badge-audio';
  if (s.includes('PPT') || s.includes('PRESENTATION')) return 'badge-ppt';
  if (s.includes('MEM')) return 'badge-memory';
  return 'badge-system';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ==========================================================================
// INTERACTIVE ACTIONS & TESTS
// ==========================================================================

async function triggerSkillTest(skillCode) {
  if (testingSkills.has(skillCode)) return;
  testingSkills.add(skillCode);
  renderDashboard(latestData || { subsystems: [] });

  try {
    const res = await fetch('/api/monitor/test_skill', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skill: skillCode })
    });
    const result = await res.json();
    console.log(`[TEST ${skillCode}] Result:`, result);
  } catch (err) {
    console.error(`[TEST ${skillCode}] Error:`, err);
  } finally {
    testingSkills.delete(skillCode);
    fetchTelemetry();
    fetchLogs();
  }
}

// Trigger VMC Expression
document.querySelectorAll('.btn-emote').forEach(btn => {
  btn.addEventListener('click', async (e) => {
    const emote = e.currentTarget.getAttribute('data-emote');
    try {
      currentExpression.textContent = `${emote} (Active)`;
      await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: `[EMOTE: ${emote}]` })
      });
    } catch (err) {
      console.warn("Emote trigger error:", err);
    }
  });
});

// Jump to errors button
if (btnJumpErrors) {
  btnJumpErrors.addEventListener('click', () => {
    document.getElementById('diagnostic-center').scrollIntoView({ behavior: 'smooth' });
  });
}

// Scan Now button
document.getElementById('btn-scan-now').addEventListener('click', () => {
  fetchTelemetry();
  fetchLogs();
});

// Camera Hardware Power Control
let isCamHardwareRunning = false;

function setCameraUIState(active) {
  isCamHardwareRunning = active;
  if (!btnCamPower) return;
  if (active) {
    btnCamPower.classList.add('active');
    if (camPowerIcon) camPowerIcon.textContent = '⏹';
    if (camPowerText) camPowerText.textContent = 'Stop Camera';
    if (visionFeedBadge) {
      visionFeedBadge.textContent = 'HARDWARE ACTIVE';
      visionFeedBadge.className = 'badge-tag badge-emerald';
    }
  } else {
    btnCamPower.classList.remove('active');
    if (camPowerIcon) camPowerIcon.textContent = '▶';
    if (camPowerText) camPowerText.textContent = 'Start Camera';
    if (visionFeedBadge) {
      visionFeedBadge.textContent = 'STANDBY';
      visionFeedBadge.className = 'badge-tag';
    }
  }
}

if (btnCamPower) {
  btnCamPower.addEventListener('click', async () => {
    btnCamPower.disabled = true;
    try {
      if (isCamHardwareRunning) {
        await fetch('/api/camera/stop', { method: 'POST' });
        setCameraUIState(false);
      } else {
        await fetch('/api/camera/start', { method: 'POST' });
        setCameraUIState(true);
      }
    } catch (err) {
      console.error("Camera toggle failed:", err);
    } finally {
      btnCamPower.disabled = false;
      setTimeout(() => {
        fetchTelemetry();
        fetchLogs();
      }, 300);
    }
  });
}

// Console Filter Tabs
document.querySelectorAll('.filter-tab').forEach(tab => {
  tab.addEventListener('click', (e) => {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    e.currentTarget.classList.add('active');
    currentFilter = e.currentTarget.getAttribute('data-filter');
    fetchLogs();
  });
});

// Console Auto-Scroll Toggle
document.getElementById('btn-toggle-autoscroll').addEventListener('click', () => {
  autoScroll = !autoScroll;
  const dot = document.getElementById('autoscroll-dot');
  dot.style.background = autoScroll ? 'var(--neon-emerald)' : 'var(--text-dim)';
});

// Clear Console
document.getElementById('btn-clear-logs').addEventListener('click', () => {
  consoleTerminal.innerHTML = '<div class="log-line log-info"><span class="log-msg">Logs cleared by user.</span></div>';
});

// Offline Fallback Handler
function handleOfflineState(err) {
  masterStatusCard.className = 'master-status-card status-error';
  masterStatusLabel.textContent = 'BACKEND OFFLINE';
  masterStatusSubtext.textContent = 'Cannot reach FastAPI server at port 8000';
  lastUpdatedText.textContent = 'Connection Lost';
  globalPulse.style.background = 'var(--neon-ruby)';
  globalPulse.style.boxShadow = '0 0 8px var(--neon-ruby)';
}

// Store last data for rapid rerender
let latestData = null;
const originalRenderDashboard = renderDashboard;
renderDashboard = function(data) {
  latestData = data;
  originalRenderDashboard(data);
};

// ==========================================================================
// START AUTO-POLL LOOP
// ==========================================================================

fetchTelemetry();
fetchLogs();

pollTimer = setInterval(() => {
  if (autoRefresh) {
    fetchTelemetry();
    fetchLogs();
  }
}, pollIntervalMs);
