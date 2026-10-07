import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Pre-defined fallback targets in case backend is loading
const FALLBACK_TARGETS = [
  {
    id: "Kepler-10",
    name: "Kepler-10 (Exoplanet Host Star)",
    tag: "Planetary System",
    expected_class: "exoplanet_transit",
    ra: 285.6794,
    dec: 50.2413,
    description: "G-type dwarf host to rocky super-Earth Kepler-10b (P=0.837d) and mini-Neptune Kepler-10c.",
    icon: "🪐",
    modalities: ["light_curve", "fits_image"],
  },
  {
    id: "KIC-9002278",
    name: "KIC-9002278 (Algol Eclipsing Binary)",
    tag: "Binary System",
    expected_class: "eclipsing_binary",
    ra: 288.7421,
    dec: 45.3129,
    description: "Detached Algol-type eclipsing binary displaying alternating primary and secondary eclipses (P=1.58d).",
    icon: "🌓",
    modalities: ["light_curve", "spectrum"],
  },
  {
    id: "KIC-9700322",
    name: "KIC-9700322 (Delta Scuti Variable Star)",
    tag: "Pulsating Star",
    expected_class: "variable_star",
    ra: 290.1250,
    dec: 46.4520,
    description: "Multi-periodic A7V pulsating variable exhibiting radial and non-radial p-mode acoustic pulsations.",
    icon: "🌟",
    modalities: ["light_curve"],
  },
  {
    id: "ASASSN-15lh",
    name: "ASASSN-15lh (Superluminous Supernova)",
    tag: "Explosive Transient",
    expected_class: "supernova",
    ra: 330.5517,
    dec: -61.6841,
    description: "Extremely energetic SLSN-I event at z=0.2326 reaching 570 billion solar luminosities.",
    icon: "💥",
    modalities: ["light_curve", "fits_image", "spectrum"],
  },
  {
    id: "GW170817-SGRB",
    name: "GW170817 / SGRB (Kilonova & GRB)",
    tag: "Multi-Messenger",
    expected_class: "gamma_ray_burst",
    ra: 197.4503,
    dec: -23.6331,
    description: "Binary neutron star merger accompanied by short gamma-ray burst and optical kilonova in NGC 4993.",
    icon: "☄️",
    modalities: ["fits_image", "spectrum"],
  },
  {
    id: "ANOMALY-OOD-99",
    name: "OOD Candidate 99 (Uncatalogued Anomaly)",
    tag: "Uncatalogued OOD",
    expected_class: "anomaly",
    ra: 180.1120,
    dec: 12.4410,
    description: "Aperiodic deep dips and stochastic flares with negative energy score exceeding anomaly threshold.",
    icon: "❓",
    modalities: ["light_curve", "fits_image", "spectrum"],
  },
];

export default function App() {
  const [targets, setTargets] = useState(FALLBACK_TARGETS);
  const [selectedTargetId, setSelectedTargetId] = useState("Kepler-10");
  const [mode, setMode] = useState("dropdown"); // "dropdown" | "upload"
  const [uploadedImageBase64, setUploadedImageBase64] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [isFitsFile, setIsFitsFile] = useState(false);
  const [customTargetId, setCustomTargetId] = useState("");
  const [fitsFilePath, setFitsFilePath] = useState("");
  const [inspectingFits, setInspectingFits] = useState(false);
  const [fitsInspectionData, setFitsInspectionData] = useState(null);
  const [includeFits, setIncludeFits] = useState(true);
  const [generateDeepSeek, setGenerateDeepSeek] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [remainingQuota, setRemainingQuota] = useState(50);
  const [moderationStats, setModerationStats] = useState({ cached_reports_count: 0, cache_hits: 0, live_calls: 0 });
  const [serverStatus, setServerStatus] = useState("Online • GPU Ready");
  const [copySuccess, setCopySuccess] = useState(false);

  const previewCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // 1. Fetch catalog targets and initial status
  useEffect(() => {
    fetch('/api/targets')
      .then(res => res.json())
      .then(data => {
        if (data.targets && data.targets.length > 0) {
          setTargets(data.targets);
        }
      })
      .catch(() => console.log("Using local catalog targets."));

    fetch('/health')
      .then(res => res.json())
      .then(data => {
        setServerStatus(`${data.status} • ${data.device}`);
      })
      .catch(() => setServerStatus("Connected (Local Mode)"));

    fetch('/api/moderation-stats')
      .then(res => res.json())
      .then(stats => setModerationStats(stats))
      .catch(() => {});
  }, []);

  // 2. Render preview curve on canvas
  const currentTarget = targets.find(t => t.id === selectedTargetId) || targets[0];

  useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !currentTarget) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 25) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Curve rendering
    const points = currentTarget.preview_curve || [];
    if (points.length === 0) return;

    const min = Math.min(...points);
    const max = Math.max(...points);
    const range = max - min || 1;

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();

    points.forEach((pt, idx) => {
      const x = (idx / (points.length - 1)) * width;
      const normalized = (pt - min) / range;
      const y = height - (normalized * (height - 24) + 12);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();

    // Subtle area gradient under curve
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
    grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();
  }, [currentTarget, mode]);

  // Handle inspecting and reading a FITS file from path
  const handleInspectAndLoadFits = async (pathOverride) => {
    const targetPath = typeof pathOverride === 'string' ? pathOverride : fitsFilePath;
    if (!targetPath || !targetPath.trim()) return;

    setInspectingFits(true);
    try {
      const res = await fetch('/api/fits/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: targetPath.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`FITS Read Error: ${err.detail || "Unable to read FITS file at specified path."}`);
        setInspectingFits(false);
        return;
      }
      const data = await res.json();
      setFitsInspectionData(data);
      setFitsFilePath(targetPath);
      setUploadedImageBase64(data.preview_data_url);
      setUploadedFileName(targetPath.split('/').pop().split('\\').pop());
      setIsFitsFile(true);
      setIncludeFits(true);
      if (data.metadata?.target_name && data.metadata.target_name !== "Custom Target") {
        setCustomTargetId(data.metadata.target_name);
      }
    } catch (err) {
      console.error(err);
      alert(`Could not connect to backend to read FITS file: ${err.message}`);
    } finally {
      setInspectingFits(false);
    }
  };

  // Handle image or FITS upload file selection
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFileName(file.name);
    const fname = file.name.toLowerCase();
    const isFits = fname.endsWith('.fits') || fname.endsWith('.fit') || fname.endsWith('.fts');
    setIsFitsFile(isFits);
    setIncludeFits(true);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const b64 = event.target.result;
      setUploadedImageBase64(b64);
      if (isFits) {
        setInspectingFits(true);
        try {
          const res = await fetch('/api/fits/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_base64: b64 }),
          });
          if (res.ok) {
            const data = await res.json();
            setFitsInspectionData(data);
            setUploadedImageBase64(data.preview_data_url);
            if (data.metadata?.target_name && data.metadata.target_name !== "Custom Target") {
              setCustomTargetId(data.metadata.target_name);
            }
          }
        } catch (err) {
          console.error(err);
        } finally {
          setInspectingFits(false);
        }
      }
    };
    reader.readAsDataURL(file);
  };

  // Generate a synthetic sample astronomical cutout using Canvas
  const generateSampleCutout = (type) => {
    const c = document.createElement('canvas');
    c.width = 224;
    c.height = 224;
    const ctx = c.getContext('2d');

    // Background sky noise
    ctx.fillStyle = '#060814';
    ctx.fillRect(0, 0, 224, 224);

    const imgData = ctx.getImageData(0, 0, 224, 224);
    for (let i = 0; i < imgData.data.length; i += 4) {
      const n = Math.floor(Math.random() * 20);
      imgData.data[i] = n;
      imgData.data[i + 1] = n;
      imgData.data[i + 2] = n + 5;
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw astronomical feature
    if (type === 'transit') {
      const rad = ctx.createRadialGradient(112, 112, 0, 112, 112, 70);
      rad.addColorStop(0, '#ffffff');
      rad.addColorStop(0.2, '#fde047');
      rad.addColorStop(0.5, '#ea580c');
      rad.addColorStop(1, 'transparent');
      ctx.fillStyle = rad;
      ctx.beginPath();
      ctx.arc(112, 112, 70, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === 'supernova') {
      // Host galaxy diffuse disk
      ctx.fillStyle = 'rgba(165, 180, 252, 0.25)';
      ctx.beginPath();
      ctx.ellipse(112, 112, 85, 35, Math.PI / 4, 0, Math.PI * 2);
      ctx.fill();

      // Bright point transient offset
      const snGrad = ctx.createRadialGradient(135, 95, 0, 135, 95, 25);
      snGrad.addColorStop(0, '#ffffff');
      snGrad.addColorStop(0.4, '#67e8f9');
      snGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = snGrad;
      ctx.beginPath();
      ctx.arc(135, 95, 25, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Dual interacting binary / merger
      ctx.fillStyle = 'rgba(236, 72, 153, 0.4)';
      ctx.beginPath();
      ctx.arc(90, 112, 35, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(134, 112, 30, 0, Math.PI * 2);
      ctx.fill();
    }

    setUploadedImageBase64(c.toDataURL('image/png'));
  };

  // Run Multi-Modal Prediction & Report Generation
  const handleRunPrediction = async () => {
    setLoading(true);

    const targetName = mode === "dropdown"
      ? selectedTargetId
      : (customTargetId.trim() || uploadedFileName.replace(/\.[^/.]+$/, "") || "Custom_FITS_Target");

    const payload = {
      target_id: targetName,
      image_base64: mode === "upload" && includeFits ? uploadedImageBase64 : null,
      fits_file_path: mode === "upload" && includeFits && fitsFilePath.trim() ? fitsFilePath.trim() : null,
      include_fits: includeFits,
      generate_deepseek_report: generateDeepSeek,
      force_fresh_report: false,
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Turn-ID': 'web_turn_session',
        },
        body: JSON.stringify(payload),
      });

      // Track remaining rate limit from response headers
      const rem = res.headers.get('x-ratelimit-remaining');
      if (rem !== null) {
        setRemainingQuota(parseInt(rem, 10));
      }

      if (!res.ok) {
        const err = await res.json();
        alert(`Error: ${err.detail || err.error || "Rate limit reached"}`);
        setLoading(false);
        return;
      }

      const data = await res.json();
      setResult(data);

      if (data.moderation_stats) {
        setModerationStats(data.moderation_stats);
      }
    } catch (err) {
      console.error(err);
      alert("Inference request failed. Make sure the API server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const copyReport = () => {
    if (!result?.scientific_report) return;
    const r = result.scientific_report;
    const md = `# AstroFoundation Scientific Report: ${r.target_id}
**Classification:** ${r.classification} (Confidence: ${(r.confidence * 100).toFixed(1)}%)
**Energy Score:** ${r.energy_score} | **Anomaly Flag:** ${r.is_anomaly ? 'YES' : 'NO'}

## Executive Summary
${r.executive_summary}

## Photometric Analysis
${r.photometric_analysis}

## Spectroscopic Analysis
${r.spectroscopic_analysis}

## Recommended Follow-up Observations
${(r.recommended_followup || []).map(f => `- ${f}`).join('\n')}
`;
    navigator.clipboard.writeText(md);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2500);
  };

  const exportMarkdown = () => {
    if (!result?.scientific_report) return;
    const r = result.scientific_report;
    const md = `# AstroFoundation Scientific Report: ${r.target_id}\n\n${r.executive_summary}\n\n### Photometric Analysis\n${r.photometric_analysis}\n\n### Spectroscopic Analysis\n${r.spectroscopic_analysis}\n`;
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AstroFoundation_${r.target_id}_Report.md`;
    a.click();
  };

  return (
    <div className="app-wrapper">
      {/* Background Cosmic Nebula */}
      <div className="cosmos-background">
        <div className="glow-orb orb-1"></div>
        <div className="glow-orb orb-2"></div>
        <div className="glow-orb orb-3"></div>
      </div>

      {/* Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="logo-badge">
            <span className="logo-icon">✨</span>
          </div>
          <div>
            <h1 className="brand-title">AstroFoundation <span className="version-tag">React v4.0</span></h1>
            <p className="brand-subtitle">Multi-Modal Foundation Model &bull; LangGraph Agent &bull; DeepSeek Report</p>
          </div>
        </div>

        <div className="header-status-group">
          <div className="status-pill">
            <span className="status-dot green pulse"></span>
            <span>{serverStatus}</span>
          </div>
          <div className="status-pill quota-pill" title="Enforcing 50 requests per turn limit">
            <span>⏱️</span>
            <span>Quota: {remainingQuota}/50 turn</span>
          </div>
          <div className="status-pill moderation-pill" title="DeepSeek Token Quota Conservation Active">
            <span>🛡️</span>
            <span>DeepSeek Moderated ({moderationStats.cache_hits} cached)</span>
          </div>
        </div>
      </header>

      {/* Main Workspace Grid */}
      <main className="workspace-grid">
        {/* Left Column: Input Selection / Upload */}
        <section className="panel input-panel">
          <div className="panel-header">
            <div className="panel-title-group">
              <span className="panel-icon">🔭</span>
              <h2>Observation Input</h2>
            </div>
            <div className="input-toggle-group">
              <button
                className={`toggle-btn ${mode === 'dropdown' ? 'active' : ''}`}
                onClick={() => setMode('dropdown')}
              >
                Catalog Presets
              </button>
              <button
                className={`toggle-btn ${mode === 'upload' ? 'active' : ''}`}
                onClick={() => setMode('upload')}
              >
                📁 Upload FITS / Image Data
              </button>
            </div>
          </div>

          {mode === 'dropdown' ? (
            <div className="tab-content">
              <div className="form-group">
                <label className="form-label" htmlFor="target-select">Select Astronomical Target</label>
                <div className="select-wrapper">
                  <select
                    id="target-select"
                    className="custom-select"
                    value={selectedTargetId}
                    onChange={(e) => setSelectedTargetId(e.target.value)}
                  >
                    {targets.map(t => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Quick Select Chips */}
              <div className="chips-container">
                <span className="chip-label">Quick select:</span>
                {targets.slice(0, 5).map(t => (
                  <button
                    key={t.id}
                    className={`target-chip ${selectedTargetId === t.id ? 'active' : ''}`}
                    onClick={() => setSelectedTargetId(t.id)}
                  >
                    {t.icon || '🪐'} {t.id}
                  </button>
                ))}
              </div>

              {/* Target Metadata Card */}
              {currentTarget && (
                <div className="target-meta-card">
                  <div className="meta-row header-row">
                    <span className="meta-name">{currentTarget.name}</span>
                    <span className="meta-badge">{currentTarget.tag || "Catalog Object"}</span>
                  </div>
                  <p className="meta-desc">{currentTarget.description}</p>
                  <div className="meta-stats-grid">
                    <div className="meta-stat">
                      <span className="stat-label">Coordinates</span>
                      <span className="stat-value font-mono">
                        {currentTarget.ra ? `${currentTarget.ra.toFixed(2)}°, ${currentTarget.dec.toFixed(2)}°` : 'N/A'}
                      </span>
                    </div>
                    <div className="meta-stat">
                      <span className="stat-label">Expected Class</span>
                      <span className="stat-value" style={{ textTransform: 'capitalize' }}>
                        {currentTarget.expected_class?.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="meta-stat">
                      <span className="stat-label">Active Modalities</span>
                      <span className="stat-value">
                        {(currentTarget.modalities || ['light_curve']).map(m => m.replace('_', ' ')).join(' • ')}
                      </span>
                    </div>
                  </div>

                  {/* Photometric Phase Curve Canvas */}
                  <div className="curve-preview-container">
                    <div className="curve-header">
                      <span className="curve-title">Photometric Phase Curve Preview</span>
                      <span className="curve-badge">Phase Folded</span>
                    </div>
                    <canvas ref={previewCanvasRef} width={420} height={90} className="canvas-plot"></canvas>
                  </div>

                  {/* FITS Source Status & Direct Navigation */}
                  <div className="catalog-fits-status-bar">
                    <div className="fits-status-info">
                      <span className="status-label">2D FITS Source:</span>
                      <span className="status-value">Auto-linked from NASA SkyView &amp; Kepler TPF</span>
                    </div>
                    <button
                      type="button"
                      className="switch-to-upload-btn"
                      onClick={() => setMode('upload')}
                    >
                      Upload Custom .FITS File &rarr;
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="tab-content">
              {/* Option 1: Direct FITS File Path Input */}
              <div className="form-group fits-path-group">
                <label className="form-label" htmlFor="fits-path-input">
                  Enter Local FITS File Path (.fits)
                </label>
                <div className="fits-path-input-row">
                  <input
                    id="fits-path-input"
                    type="text"
                    className="custom-text-input font-mono"
                    placeholder="e.g. data/sample_images/kepler_10_transit.fits or C:/path/to/target.fits"
                    value={fitsFilePath}
                    onChange={(e) => setFitsFilePath(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleInspectAndLoadFits(); }}
                  />
                  <button
                    type="button"
                    className="load-fits-btn"
                    onClick={() => handleInspectAndLoadFits()}
                    disabled={inspectingFits || !fitsFilePath.trim()}
                  >
                    {inspectingFits ? (
                      <>
                        <span className="btn-spinner sm"></span>
                        <span>Reading...</span>
                      </>
                    ) : (
                      <>
                        <span>🔍</span>
                        <span>Read FITS</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Quick Sample FITS Buttons */}
              <div className="quick-fits-presets">
                <span className="chip-label">Quick sample FITS files:</span>
                <button
                  type="button"
                  className="quick-fits-chip"
                  onClick={() => handleInspectAndLoadFits("data/sample_images/kepler_10_transit.fits")}
                >
                  🪐 Kepler-10 FITS
                </button>
                <button
                  type="button"
                  className="quick-fits-chip"
                  onClick={() => handleInspectAndLoadFits("data/sample_images/sn2011fe_supernova.fits")}
                >
                  💥 SN 2011fe FITS
                </button>
                <button
                  type="button"
                  className="quick-fits-chip"
                  onClick={() => handleInspectAndLoadFits("data/sample_images/beta_lyrae_binary.fits")}
                >
                  🌀 Beta Lyrae FITS
                </button>
                <button
                  type="button"
                  className="quick-fits-chip"
                  onClick={() => handleInspectAndLoadFits("data/sample_images/ngc4993_grb_kilonova.fits")}
                >
                  ☄️ NGC 4993 FITS
                </button>
              </div>

              {/* FITS Inspection Card (Rendered if FITS is loaded) */}
              {fitsInspectionData && (
                <div className="fits-inspection-card">
                  <div className="fits-inspection-header">
                    <span className="fits-status-badge">✅ Valid FITS Loaded ({fitsInspectionData.metadata.hdu_count} HDU)</span>
                    <button
                      type="button"
                      className="clear-fits-btn"
                      onClick={() => {
                        setFitsInspectionData(null);
                        setUploadedImageBase64(null);
                        setUploadedFileName("");
                        setFitsFilePath("");
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="fits-inspection-body">
                    <div className="fits-thumbnail-box">
                      <img src={fitsInspectionData.preview_data_url} alt="FITS Visual Heatmap" />
                      <span className="thumb-caption">ArcSinh Visual Heatmap</span>
                    </div>
                    <div className="fits-meta-table">
                      <div className="meta-item">
                        <span className="meta-k">Target:</span>
                        <span className="meta-v">{fitsInspectionData.metadata.target_name}</span>
                      </div>
                      <div className="meta-item">
                        <span className="meta-k">Dimensions:</span>
                        <span className="meta-v font-mono">{fitsInspectionData.pixel_stats.original_shape.join(' × ')}</span>
                      </div>
                      <div className="meta-item">
                        <span className="meta-k">Telescope:</span>
                        <span className="meta-v">{fitsInspectionData.metadata.telescope}</span>
                      </div>
                      <div className="meta-item">
                        <span className="meta-k">Pixel Min/Max:</span>
                        <span className="meta-v font-mono">{fitsInspectionData.pixel_stats.min.toFixed(0)} &bull; {fitsInspectionData.pixel_stats.max.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Option 2: Drag & Drop File Upload */}
              <div className="divider-or"><span>OR UPLOAD FILE</span></div>

              <div
                className="upload-dropzone"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".fits,.fit,.fts,image/*"
                  className="file-input-hidden"
                  onChange={handleFileChange}
                />
                {!uploadedImageBase64 ? (
                  <div>
                    <span className="dropzone-icon">📁</span>
                    <p className="dropzone-text"><strong>Drag &amp; drop your .FITS file or optical image</strong> here</p>
                    <p className="dropzone-sub">Supports Astronomical FITS (.fits, .fit) and Optical Cutouts (PNG, JPEG)</p>
                    <button type="button" className="browse-files-btn">Browse Local Files</button>
                  </div>
                ) : (
                  <div className="image-preview-wrapper" onClick={(e) => e.stopPropagation()}>
                    <img src={uploadedImageBase64} alt="Astronomical Cutout Preview" />
                    <button
                      className="remove-img-btn"
                      onClick={() => {
                        setUploadedImageBase64(null);
                        setUploadedFileName("");
                        setIsFitsFile(false);
                        setFitsInspectionData(null);
                      }}
                      title="Remove file"
                    >
                      &times;
                    </button>
                  </div>
                )}
              </div>

              {/* Optional Custom Target Identifier */}
              <div className="form-group" style={{ marginTop: '0.85rem' }}>
                <label className="form-label" htmlFor="custom-target-id">Custom Target Identifier (Optional)</label>
                <input
                  id="custom-target-id"
                  type="text"
                  className="custom-text-input"
                  placeholder="e.g. TIC-25155310, KIC-9002278, or SN-2026xyz"
                  value={customTargetId}
                  onChange={(e) => setCustomTargetId(e.target.value)}
                />
              </div>
            </div>
          )}

          {/* DeepSeek Moderation Controls & Action */}
          <div className="action-footer">
            <div className="moderation-controls">
              {/* Optional 2D FITS Calculation Toggle */}
              <div className="modality-toggle-card">
                <label className="toggle-checkbox-label">
                  <input
                    type="checkbox"
                    checked={includeFits}
                    onChange={(e) => setIncludeFits(e.target.checked)}
                  />
                  <span className="custom-checkbox"></span>
                  <div className="toggle-text-block">
                    <div className="toggle-header-row">
                      <span className="toggle-title">Include 2D FITS Vision Calculation (Optional)</span>
                      <span className={`modality-status-pill ${includeFits ? 'active' : 'bypassed'}`}>
                        {includeFits ? 'Vision Active' : 'Optional / Bypassed'}
                      </span>
                    </div>
                    <span className="toggle-desc">
                      {includeFits
                        ? "2D FITS vision backbone active (arcsinh + MAD feature map). Uncheck if only light curve or spectra are available."
                        : "FITS calculation bypassed: Model predicts purely from 1D Light Curves & Optical Spectra."}
                    </span>
                  </div>
                </label>
              </div>

              <label className="toggle-checkbox-label">
                <input
                  type="checkbox"
                  checked={generateDeepSeek}
                  onChange={(e) => setGenerateDeepSeek(e.target.checked)}
                />
                <span className="custom-checkbox"></span>
                <span>Generate DeepSeek Scientific Report</span>
              </label>
              <div className="moderation-notice">
                <span>🛡️</span>
                <span>Smart Moderation Active: Caches repeated queries to protect API quota</span>
              </div>
            </div>

            <button
              className="run-inference-btn"
              onClick={handleRunPrediction}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="btn-spinner"></span>
                  <span>Executing Multi-Modal Neural Model...</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Run Multi-Modal Prediction</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* Right Column: Prediction & DeepSeek Report */}
        <section className="panel results-panel">
          <div className="panel-header">
            <div className="panel-title-group">
              <span className="panel-icon">📊</span>
              <h2>Multi-Modal Prediction &amp; AI Analysis</h2>
            </div>
            {result && (
              <div className="badge-row">
                <span className="pill-chip">Target: {result.target_id}</span>
                <span className="pill-chip latency-chip">{result.execution_time_seconds}s</span>
              </div>
            )}
          </div>

          {!result ? (
            <div className="placeholder-state">
              <div className="placeholder-icon">🔭</div>
              <h3>Awaiting Astronomical Target</h3>
              <p>Select a catalog target from the presets or upload a custom image cutout, then click <strong>Run Multi-Modal Prediction</strong>.</p>
            </div>
          ) : (
            <div>
              {/* Classification Hero */}
              <div className="classification-hero">
                <div>
                  <span className="hero-target-label">{result.target_id}</span>
                  <div className="hero-class-group">
                    <span className="hero-class-name">{result.classification.replace('_', ' ')}</span>
                    <span className={`hero-anomaly-tag ${result.is_anomaly ? 'anomaly' : 'in-distribution'}`}>
                      {result.is_anomaly ? '⚠️ Anomaly (OOD)' : '✅ In-Distribution'}
                    </span>
                    <span className={`hero-modality-tag ${result.fits_calculation_mode === 'ACTIVE' ? 'fits-active' : 'fits-bypassed'}`}>
                      {result.fits_calculation_mode === 'ACTIVE' ? '🖼️ FITS 2D Active' : '⚡ FITS Optional (Bypassed)'}
                    </span>
                  </div>
                </div>
                <div className="hero-right">
                  <div className="metric-block">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-val">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-block">
                    <span className="metric-label">Energy Score</span>
                    <span className="metric-val font-mono">{result.energy_score.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Probability Distribution Bars */}
              <div className="card">
                <div className="card-title">Event Class Posterior Probabilities</div>
                <div className="prob-list">
                  {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                    <div key={cls} className="prob-item">
                      <div className="prob-row-header">
                        <span className="prob-class-name">{cls.replace('_', ' ')}</span>
                        <span className="prob-val-text font-mono">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="prob-bar-track">
                        <div
                          className="prob-bar-fill"
                          style={{ width: `${Math.max(2, prob * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Explainable AI (XAI): Grad-CAM & SHAP Feature Attributions */}
              {result.xai_attributions && (
                <div className="card xai-card">
                  <div className="card-title-row">
                    <div className="card-title">Explainable AI (XAI): Grad-CAM &amp; SHAP Attributions</div>
                    <span className="agent-consensus-pill xai-pill">Captum &bull; SHAP &bull; Grad-CAM</span>
                  </div>
                  <div className="xai-contributions-row">
                    <div className="xai-label-row">
                      <span className="stat-label">Multi-Modal Shapley Attribution Weights</span>
                    </div>
                    <div className="modality-bar-wrapper">
                      <div
                        className="modality-bar-segment lc"
                        style={{ width: `${result.xai_attributions.modality_contributions.light_curve_pct}%` }}
                      >
                        Light Curve {result.xai_attributions.modality_contributions.light_curve_pct}%
                      </div>
                      <div
                        className="modality-bar-segment img"
                        style={{ width: `${result.xai_attributions.modality_contributions.fits_image_pct}%` }}
                      >
                        FITS {result.xai_attributions.modality_contributions.fits_image_pct}%
                      </div>
                      <div
                        className="modality-bar-segment spec"
                        style={{ width: `${result.xai_attributions.modality_contributions.spectra_pct}%` }}
                      >
                        Spectrum {result.xai_attributions.modality_contributions.spectra_pct}%
                      </div>
                    </div>
                  </div>

                  <div className="salient-features-list">
                    <div className="xai-label-row">
                      <span className="stat-label">Astrophysical Salient Features (1D/2D Grad-CAM Peak Detection)</span>
                    </div>
                    <div className="features-tags-group">
                      {result.xai_attributions.salient_features.map((feat, idx) => (
                        <div key={idx} className="feature-tag">
                          <span className="feature-dot" data-mod={feat.modality}></span>
                          <span className="feature-name">{feat.name}</span>
                          <span className="feature-score font-mono">{(feat.importance * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* LangGraph Agent Database Cross-Match */}
              <div className="card">
                <div className="card-title-row">
                  <div className="card-title">LangGraph Autonomous Agent Verification</div>
                  <span className="agent-consensus-pill">{result.agent_consensus}</span>
                </div>
                <div className="crossmatch-grid">
                  <div className="crossmatch-item">
                    <span className="db-badge">SIMBAD Catalog</span>
                    <span className="db-val font-mono">
                      {result.agent_output?.simbad_match?.found ? `Confirmed: ${result.agent_output.simbad_match.main_id}` : 'No SIMBAD Record'}
                    </span>
                  </div>
                  <div className="crossmatch-item">
                    <span className="db-badge">NASA/IPAC NED</span>
                    <span className="db-val font-mono">
                      {result.agent_output?.ned_match?.found ? `Extragalactic: z=${result.agent_output.ned_match.redshift}` : 'Galactic Source'}
                    </span>
                  </div>
                  <div className="crossmatch-item">
                    <span className="db-badge">NASA ADS Papers</span>
                    <span className="db-val font-mono">
                      {result.agent_output?.ads_match?.paper_count ? `${result.agent_output.ads_match.paper_count} Papers Found` : 'Catalog Cross-Matched'}
                    </span>
                  </div>
                </div>
              </div>

              {/* DeepSeek Publication Scientific Report */}
              {result.scientific_report && (
                <div className="card report-card">
                  <div className="report-header-row">
                    <div className="card-title report-title">
                      <span>DeepSeek Scientific Analysis Report</span>
                    </div>
                    <div className="deepseek-badge-group">
                      <span className={`urgency-pill urgency-${(result.scientific_report.observational_urgency || 'routine').toLowerCase()}`}>
                        {result.scientific_report.observational_urgency === 'IMMEDIATE_TOO_24H'
                          ? '🚨 ToO < 24h'
                          : result.scientific_report.observational_urgency === 'HIGH_PRIORITY_14D'
                          ? '⚡ Priority < 14d'
                          : '📅 Routine'}
                      </span>
                      <span className="deepseek-source-badge">
                        {result.scientific_report.metadata?.deepseek_source === 'live_deepseek'
                          ? '⚡ DeepSeek AI (Live)'
                          : '💾 DeepSeek (Cached)'}
                      </span>
                      <span className="deepseek-cache-badge">0 Extra Tokens</span>
                    </div>
                  </div>

                  <div className="report-section">
                    <h4 className="section-heading">Executive Summary</h4>
                    <p className="section-content">{result.scientific_report.executive_summary}</p>
                  </div>

                  <div className="report-section-split">
                    <div className="report-section">
                      <h4 className="section-heading">Photometric Analysis</h4>
                      <p className="section-content">{result.scientific_report.photometric_analysis}</p>
                    </div>
                    <div className="report-section">
                      <h4 className="section-heading">Spectroscopic Analysis</h4>
                      <p className="section-content">{result.scientific_report.spectroscopic_analysis}</p>
                    </div>
                  </div>

                  {/* Derived Physical Parameters */}
                  {result.scientific_report.physical_parameters && Object.keys(result.scientific_report.physical_parameters).length > 0 && (
                    <div className="report-section">
                      <h4 className="section-heading">Derived Astrophysical Physical Parameters</h4>
                      <div className="phys-params-grid">
                        {Object.entries(result.scientific_report.physical_parameters).map(([key, val]) => (
                          <div key={key} className="phys-param-item">
                            <span className="phys-key">{key.replace(/_/g, ' ')}</span>
                            <span className="phys-val font-mono">{String(val)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* False Positive & Contamination Forensics */}
                  {result.scientific_report.false_positive_risk && (
                    <div className="report-section">
                      <div className="section-heading-row">
                        <h4 className="section-heading">False Positive &amp; Contamination Forensics</h4>
                        <span className={`risk-level-badge risk-${(result.scientific_report.false_positive_risk.risk_level || 'low').toLowerCase()}`}>
                          Risk: {result.scientific_report.false_positive_risk.risk_level || 'LOW'}
                        </span>
                      </div>
                      <div className="fp-risk-grid">
                        {Object.entries(result.scientific_report.false_positive_risk)
                          .filter(([k]) => k !== 'risk_level')
                          .map(([key, val]) => (
                            <div key={key} className="fp-risk-item">
                              <span className="stat-label">{key.replace(/_/g, ' ')}</span>
                              <span className="fp-val">{String(val)}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {result.scientific_report.recommended_followup?.length > 0 && (
                    <div className="report-section">
                      <h4 className="section-heading">Telescope Follow-up Recommendations</h4>
                      <ul className="followup-list">
                        {result.scientific_report.recommended_followup.map((f, i) => (
                          <li key={i}>{f}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Community Alert & Telegram Bulletin Draft */}
                  {result.scientific_report.community_alert_draft && (
                    <div className="report-section alert-draft-section">
                      <div className="section-heading-row">
                        <h4 className="section-heading">Astronomer's Telegram (ATel / GCN) Bulletin Draft</h4>
                        <button
                          className="copy-alert-btn"
                          onClick={() => {
                            navigator.clipboard.writeText(result.scientific_report.community_alert_draft);
                            alert("Copied ATel / GCN Bulletin to clipboard!");
                          }}
                        >
                          📋 Copy Bulletin
                        </button>
                      </div>
                      <pre className="alert-pre font-mono">{result.scientific_report.community_alert_draft}</pre>
                    </div>
                  )}

                  <div className="report-footer-actions">
                    <button className="copy-report-btn" onClick={copyReport}>
                      {copySuccess ? '✓ Copied to Clipboard!' : '📋 Copy Full Scientific Report'}
                    </button>
                    <button className="export-report-btn" onClick={exportMarkdown}>
                      📥 Export Markdown
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div>
          <span>AstroFoundation &bull; Multi-Modal Astronomical Deep Learning System</span>
        </div>
        <div className="token-saver-stat">
          DeepSeek Quota Protection: Active &bull; {moderationStats.cache_hits} Queries Conserved
        </div>
      </footer>
    </div>
  );
}
