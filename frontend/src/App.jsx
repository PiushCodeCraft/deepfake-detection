import React, { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:5000";

export default function App() {
  const [activeTab, setActiveTab] = useState("detector"); // 'detector' | 'benchmarks' | 'history'
  const [systemStatus, setSystemStatus] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [samples, setSamples] = useState([]);
  const fileInputRef = useRef(null);

  // Fetch initial status, metrics, samples
  useEffect(() => {
    fetchSystemStatus();
    fetchMetrics();
    fetchSamples();
    fetchHistory();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setSystemStatus(data);
    } catch {
      setSystemStatus({ backend: "Offline", ml_model: "Unknown", database: "Offline" });
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch {}
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
      }
    } catch {}
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch {}
  };

  // Handle file select
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setErrorMsg("");
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // Handle drag and drop
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setResult(null);
      setErrorMsg("");
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // Load sample image
  const handleSelectSample = async (sampleName) => {
    setAnalyzing(true);
    setErrorMsg("");
    setResult(null);
    setSelectedFile(null);
    setImagePreview(`${API_BASE}/api/sample-images/${sampleName.startsWith("fake_") || sampleName.startsWith("real_") ? (sampleName.includes("original") ? sampleName : `batch/${sampleName}`) : `batch/${sampleName}`}`);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: sampleName })
      });
      const data = await res.json();
      if (data.status === "error") {
        setErrorMsg(data.message || "Face detection failed for this sample.");
      } else {
        setResult(data);
        fetchHistory();
      }
    } catch (err) {
      setErrorMsg("Failed to connect to forensic backend: " + err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  // Run analysis on uploaded file
  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    setErrorMsg("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "error") {
        setErrorMsg(data.message || "Analysis error.");
      } else {
        setResult(data);
        fetchHistory();
      }
    } catch (err) {
      setErrorMsg("Backend communication failure: " + err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top Navigation */}
      <header style={{
        borderBottom: "1px solid var(--border-subtle)",
        background: "rgba(11, 15, 25, 0.8)",
        backdropFilter: "blur(12px)",
        padding: "16px 32px",
        position: "sticky",
        top: 0,
        zIndex: 50
      }}>
        <div style={{ maxWidth: "1280px", margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{
              width: "38px",
              height: "38px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #06b6d4, #3b82f6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 15px rgba(6, 182, 212, 0.4)"
            }}>
              <span style={{ fontSize: "1.2rem", fontWeight: "bold" }}>🔍</span>
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <h1 style={{ fontSize: "1.25rem", fontWeight: "800", letterSpacing: "-0.5px" }}>
                  Deep<span style={{ color: "#38bdf8" }}>Trace</span>
                </h1>
                <span style={{
                  fontSize: "0.65rem",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: "rgba(56, 189, 248, 0.15)",
                  color: "#38bdf8",
                  fontFamily: "var(--font-mono)",
                  fontWeight: "600"
                }}>v2.0 • 22 FEATURES</span>
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Forensic Facial Landmark Geometry & Symmetry Analyzer
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div style={{ display: "flex", gap: "8px", background: "rgba(255, 255, 255, 0.04)", padding: "4px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
            <button
              onClick={() => setActiveTab("detector")}
              style={{
                background: activeTab === "detector" ? "rgba(56, 189, 248, 0.15)" : "transparent",
                color: activeTab === "detector" ? "#38bdf8" : "var(--text-muted)",
                border: activeTab === "detector" ? "1px solid rgba(56, 189, 248, 0.3)" : "none",
                padding: "8px 16px",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "0.85rem",
                transition: "all 0.2s"
              }}
            >
              Forensic Scanner
            </button>
            <button
              onClick={() => setActiveTab("benchmarks")}
              style={{
                background: activeTab === "benchmarks" ? "rgba(56, 189, 248, 0.15)" : "transparent",
                color: activeTab === "benchmarks" ? "#38bdf8" : "var(--text-muted)",
                border: activeTab === "benchmarks" ? "1px solid rgba(56, 189, 248, 0.3)" : "none",
                padding: "8px 16px",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "0.85rem",
                transition: "all 0.2s"
              }}
            >
              Model Metrics & ROC
            </button>
            <button
              onClick={() => setActiveTab("history")}
              style={{
                background: activeTab === "history" ? "rgba(56, 189, 248, 0.15)" : "transparent",
                color: activeTab === "history" ? "#38bdf8" : "var(--text-muted)",
                border: activeTab === "history" ? "1px solid rgba(56, 189, 248, 0.3)" : "none",
                padding: "8px 16px",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "0.85rem",
                transition: "all 0.2s"
              }}
            >
              Scan History
            </button>
          </div>

          {/* System Status Pill */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.8rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
            <span style={{
              display: "inline-block",
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: systemStatus?.backend === "Online" ? "#10b981" : "#ef4444",
              boxShadow: systemStatus?.backend === "Online" ? "0 0 8px #10b981" : "none"
            }} />
            <span>{systemStatus?.backend === "Online" ? "ML Engine Online" : "Connecting..."}</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ maxWidth: "1280px", margin: "0 auto", padding: "32px 24px", width: "100%", flex: 1 }}>

        {/* TAB 1: FORENSIC DETECTOR */}
        {activeTab === "detector" && (
          <div>
            {/* Quick Hero Banner */}
            <div style={{ marginBottom: "28px" }}>
              <h2 style={{ fontSize: "1.75rem", fontWeight: "700", marginBottom: "8px" }}>
                Deepfake Biometric Integrity Analysis
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", maxWidth: "800px", lineHeight: "1.5" }}>
                Upload an image or pick a test subject to extract 478 MediaPipe facial landmarks, quantify bilateral facial asymmetry, and evaluate 22 anatomical proportion ratios with explainable ML.
              </p>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: "28px" }}>
              {/* Left Column: Image Input & Scanner Preview */}
              <div>
                <div className="glass-panel" style={{ padding: "24px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                    <h3 style={{ fontSize: "1rem", fontWeight: "600" }}>Input Source</h3>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>JPEG, PNG</span>
                  </div>

                  {/* Drag and Drop Zone */}
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      border: "2px dashed rgba(56, 189, 248, 0.3)",
                      borderRadius: "12px",
                      padding: "24px",
                      textAlign: "center",
                      cursor: "pointer",
                      background: "rgba(56, 189, 248, 0.02)",
                      transition: "all 0.2s ease"
                    }}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept="image/*"
                      style={{ display: "none" }}
                    />
                    <div style={{ fontSize: "2rem", marginBottom: "8px" }}>📷</div>
                    <p style={{ fontWeight: "600", fontSize: "0.9rem", color: "var(--text-main)" }}>
                      Click to browse or drag & drop image
                    </p>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      High resolution portrait recommended
                    </p>
                  </div>

                  {/* Preset 1-Click Samples */}
                  <div style={{ marginTop: "20px" }}>
                    <p style={{ fontSize: "0.8rem", color: "var(--text-dim)", marginBottom: "10px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                      Instant Verification Presets
                    </p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                      <button
                        className="btn-sample"
                        onClick={() => handleSelectSample("real_original.jpg")}
                      >
                        ✅ Real Subject (Original)
                      </button>
                      <button
                        className="btn-sample"
                        onClick={() => handleSelectSample("fake_original.jpg")}
                      >
                        ⚠️ Fake Subject (GAN Artifact)
                      </button>
                      <button
                        className="btn-sample"
                        onClick={() => handleSelectSample("edge_case_no_face_landscape.jpg")}
                      >
                        🌲 Edge Case (Landscape)
                      </button>
                    </div>
                  </div>

                  {/* Image Preview & Scanner Overlay */}
                  {imagePreview && (
                    <div style={{ marginTop: "24px" }}>
                      <div style={{
                        position: "relative",
                        borderRadius: "12px",
                        overflow: "hidden",
                        border: "1px solid var(--border-glow)",
                        background: "#000",
                        maxHeight: "360px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center"
                      }}>
                        <img
                          src={imagePreview}
                          alt="Subject Preview"
                          style={{ width: "100%", maxHeight: "360px", objectFit: "contain", display: "block" }}
                        />
                        <div className="reticle-grid" />
                        {analyzing && <div className="scanner-laser" />}
                        <div style={{
                          position: "absolute",
                          bottom: "8px",
                          left: "8px",
                          background: "rgba(0,0,0,0.7)",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "0.7rem",
                          fontFamily: "var(--font-mono)",
                          color: "#38bdf8"
                        }}>
                          {analyzing ? "⚡ COMPUTING LANDMARKS..." : "STANDBY"}
                        </div>
                      </div>

                      {/* Trigger Analyze button if user manually uploaded */}
                      {selectedFile && (
                        <button
                          className="btn-primary"
                          onClick={handleAnalyze}
                          disabled={analyzing}
                          style={{ width: "100%", marginTop: "16px", padding: "12px" }}
                        >
                          {analyzing ? "Extracting 478 Landmarks & Analyzing..." : "Run Forensic Biometric Scan"}
                        </button>
                      )}
                    </div>
                  )}

                  {/* Error Notification */}
                  {errorMsg && (
                    <div style={{
                      marginTop: "16px",
                      padding: "12px 16px",
                      borderRadius: "10px",
                      background: "rgba(239, 68, 68, 0.1)",
                      border: "1px solid rgba(239, 68, 68, 0.3)",
                      color: "#f87171",
                      fontSize: "0.85rem"
                    }}>
                      ⚠️ {errorMsg}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Forensic Results Dashboard */}
              <div>
                {result ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                    {/* Verdict Card */}
                    <div className="glass-panel" style={{
                      padding: "24px",
                      border: result.prediction === "Real" ? "1px solid rgba(16, 185, 129, 0.5)" : "1px solid rgba(239, 68, 68, 0.5)",
                      background: result.prediction === "Real" ? "rgba(16, 185, 129, 0.05)" : "rgba(239, 68, 68, 0.05)"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div>
                          <span style={{
                            fontSize: "0.75rem",
                            textTransform: "uppercase",
                            letterSpacing: "1px",
                            fontFamily: "var(--font-mono)",
                            color: result.prediction === "Real" ? "#34d399" : "#f87171"
                          }}>
                            FORENSIC CLASSIFICATION VERDICT
                          </span>
                          <h3 style={{
                            fontSize: "2rem",
                            fontWeight: "800",
                            marginTop: "4px",
                            color: result.prediction === "Real" ? "#34d399" : "#f87171"
                          }}>
                            {result.prediction === "Real" ? "AUTHENTIC (REAL)" : "MANIPULATED (FAKE)"}
                          </h3>
                        </div>
                        <div style={{
                          padding: "12px 20px",
                          borderRadius: "12px",
                          textAlign: "center",
                          background: result.prediction === "Real" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                          border: result.prediction === "Real" ? "1px solid rgba(16, 185, 129, 0.3)" : "1px solid rgba(239, 68, 68, 0.3)"
                        }}>
                          <div style={{ fontSize: "1.75rem", fontWeight: "800", fontFamily: "var(--font-mono)" }}>
                            {result.confidence}%
                          </div>
                          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>CONFIDENCE</div>
                        </div>
                      </div>

                      {/* Probability Split Bar */}
                      <div style={{ marginTop: "20px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "6px", fontFamily: "var(--font-mono)" }}>
                          <span style={{ color: "#34d399" }}>Real: {result.probabilities?.Real || 0}%</span>
                          <span style={{ color: "#f87171" }}>Fake: {result.probabilities?.Fake || 0}%</span>
                        </div>
                        <div style={{ height: "10px", borderRadius: "5px", background: "rgba(255,255,255,0.1)", overflow: "hidden", display: "flex" }}>
                          <div style={{ width: `${result.probabilities?.Real || 0}%`, background: "#10b981", transition: "width 0.6s ease" }} />
                          <div style={{ width: `${result.probabilities?.Fake || 0}%`, background: "#ef4444", transition: "width 0.6s ease" }} />
                        </div>
                      </div>

                      {/* Explanation Narrative */}
                      <div style={{
                        marginTop: "20px",
                        padding: "16px",
                        borderRadius: "10px",
                        background: "rgba(0, 0, 0, 0.3)",
                        border: "1px solid var(--border-subtle)"
                      }}>
                        <h4 style={{ fontSize: "0.85rem", fontWeight: "600", color: "#38bdf8", marginBottom: "6px" }}>
                          Forensic Reasoning
                        </h4>
                        <p style={{ fontSize: "0.85rem", lineHeight: "1.5", color: "var(--text-main)" }}>
                          {result.explanation}
                        </p>
                      </div>
                    </div>

                    {/* Facial Symmetry Analysis Grid */}
                    <div className="glass-panel" style={{ padding: "20px" }}>
                      <h4 style={{ fontSize: "0.95rem", fontWeight: "600", marginBottom: "14px" }}>
                        Bilateral Facial Symmetry Metrics
                      </h4>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                        {[
                          { label: "Eyebrow Symmetry", val: result.symmetry_metrics?.eyebrow_symmetry },
                          { label: "Eye Aspect Symmetry", val: result.symmetry_metrics?.eye_aspect_symmetry },
                          { label: "Eye Width Symmetry", val: result.symmetry_metrics?.eye_width_symmetry },
                          { label: "Eye Height Symmetry", val: result.symmetry_metrics?.eye_height_symmetry }
                        ].map((s, idx) => {
                          const pct = (s.val * 100).toFixed(1);
                          const isHigh = s.val > 0.08;
                          return (
                            <div key={idx} style={{
                              padding: "12px",
                              borderRadius: "10px",
                              background: "rgba(255, 255, 255, 0.02)",
                              border: "1px solid var(--border-subtle)"
                            }}>
                              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{s.label}</div>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: "4px" }}>
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: "700", color: isHigh ? "#f87171" : "#34d399" }}>
                                  {pct}%
                                </span>
                                <span style={{ fontSize: "0.65rem", color: isHigh ? "#f87171" : "var(--text-dim)" }}>
                                  {isHigh ? "High Asymmetry" : "Normal"}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Top Predictive Biometric Features */}
                    <div className="glass-panel" style={{ padding: "20px" }}>
                      <h4 style={{ fontSize: "0.95rem", fontWeight: "600", marginBottom: "14px" }}>
                        Influential Biometric Landmarks for Decision
                      </h4>
                      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                        {result.top_features?.slice(0, 6).map((tf, i) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.85rem" }}>
                            <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{tf.feature}</span>
                            <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
                              <span style={{ fontFamily: "var(--font-mono)", color: "#38bdf8" }}>{tf.value.toFixed(4)}</span>
                              <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", width: "65px", textAlign: "right" }}>
                                {(tf.importance * 100).toFixed(1)}% wt
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Empty state placeholder */
                  <div className="glass-panel" style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    minHeight: "400px"
                  }}>
                    <div style={{
                      width: "64px",
                      height: "64px",
                      borderRadius: "50%",
                      background: "rgba(56, 189, 248, 0.1)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "1.8rem",
                      marginBottom: "16px"
                    }}>
                      ⚡
                    </div>
                    <h3 style={{ fontSize: "1.2rem", fontWeight: "600", marginBottom: "8px" }}>
                      Awaiting Image Scan
                    </h3>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "340px", lineHeight: "1.5" }}>
                      Upload an image or choose one of the preset subjects on the left to inspect 478 MediaPipe landmarks and 22 geometric ratios.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: MODEL BENCHMARKS & ROC */}
        {activeTab === "benchmarks" && (
          <div>
            <h2 style={{ fontSize: "1.75rem", fontWeight: "700", marginBottom: "8px" }}>
              Machine Learning Architecture & Validation
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginBottom: "28px" }}>
              DeepTrace employs a 22-dimensional facial geometry feature space derived from MediaPipe FaceLandmarker (478 coordinates), classified via a balanced Random Forest ensemble.
            </p>

            {/* Metric KPI Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px", marginBottom: "28px" }}>
              {[
                { title: "5-Fold Mean Accuracy", val: metrics ? `${(metrics.cross_validation_5fold.cv_accuracy_mean * 100).toFixed(2)}%` : "83.71%", note: "±3.19% across folds" },
                { title: "5-Fold ROC-AUC Score", val: metrics ? `${metrics.cross_validation_5fold.cv_roc_auc_mean.toFixed(4)}` : "0.9122", note: "±0.0196 across folds" },
                { title: "Held-out Test Precision", val: metrics ? `${(metrics.single_test_split.precision * 100).toFixed(2)}%` : "82.41%", note: "On 390 test samples" },
                { title: "Held-out Test Recall", val: metrics ? `${(metrics.single_test_split.recall * 100).toFixed(2)}%` : "83.67%", note: "On 390 test samples" }
              ].map((m, idx) => (
                <div key={idx} className="glass-panel" style={{ padding: "20px" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>{m.title}</div>
                  <div style={{ fontSize: "1.8rem", fontWeight: "800", fontFamily: "var(--font-mono)", color: "#38bdf8", marginTop: "6px" }}>
                    {m.val}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "4px" }}>{m.note}</div>
                </div>
              ))}
            </div>

            {/* Pipeline Architecture Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px" }}>
              <div className="glass-panel" style={{ padding: "24px" }}>
                <h3 style={{ fontSize: "1.1rem", fontWeight: "600", marginBottom: "16px" }}>
                  Active 22-Feature Biometric Vector
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", fontSize: "0.8rem", fontFamily: "var(--font-mono)" }}>
                  {[
                    "left_eye_aspect_ratio", "right_eye_aspect_ratio",
                    "nose_width", "mouth_width",
                    "face_height", "nose_to_mouth",
                    "eyebrow_distance", "left_eye_height",
                    "eyebrow_symmetry", "right_eye_height",
                    "face_aspect_ratio", "nose_mouth_ratio",
                    "face_width", "right_eye_width",
                    "eye_distance", "nose_angle",
                    "nose_width_ratio", "left_eye_width",
                    "eye_width_symmetry", "eye_aspect_symmetry",
                    "mouth_height", "eye_height_symmetry"
                  ].map((feat, i) => (
                    <div key={i} style={{
                      padding: "8px 12px",
                      borderRadius: "6px",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid var(--border-subtle)",
                      color: "var(--text-muted)"
                    }}>
                      <span style={{ color: "#38bdf8", marginRight: "6px" }}>#{i + 1}</span> {feat}
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: "24px" }}>
                <h3 style={{ fontSize: "1.1rem", fontWeight: "600", marginBottom: "16px" }}>
                  Key Architectural Decisions
                </h3>
                <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "14px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                  <li style={{ display: "flex", gap: "10px" }}>
                    <span>🛡️</span>
                    <div>
                      <strong>Removal of mouth_aspect_ratio:</strong> Eliminated unstable closed-mouth landmark denominator explosions (landmarks 13 & 14 distance approaching zero).
                    </div>
                  </li>
                  <li style={{ display: "flex", gap: "10px" }}>
                    <span>📐</span>
                    <div>
                      <strong>Outer Lip Boundary Definition:</strong> Updated mouth height calculation to landmarks 0 (upper vermilion) and 17 (lower vermilion) for robust vertical lip measurement.
                    </div>
                  </li>
                  <li style={{ display: "flex", gap: "10px" }}>
                    <span>⚡</span>
                    <div>
                      <strong>19.0 ms Edge Inference:</strong> Ultra-fast MediaPipe landmarker execution enabling real-time analysis in web environments.
                    </div>
                  </li>
                  <li style={{ display: "flex", gap: "10px" }}>
                    <span>📈</span>
                    <div>
                      <strong>ROC-AUC of 0.912:</strong> High threshold-independent discrimination between authentic and synthetically generated imagery.
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: SCAN HISTORY */}
        {activeTab === "history" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
              <div>
                <h2 style={{ fontSize: "1.75rem", fontWeight: "700" }}>Forensic Scan Audit Log</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                  Recorded image analyses stored in PostgreSQL with biometric explainability metadata.
                </p>
              </div>
              <button className="btn-sample" onClick={fetchHistory}>
                🔄 Refresh Logs
              </button>
            </div>

            <div className="glass-panel" style={{ overflow: "hidden" }}>
              {history.length > 0 ? (
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(255,255,255,0.02)" }}>
                      <th style={{ padding: "14px 18px", color: "var(--text-muted)" }}>Subject / File</th>
                      <th style={{ padding: "14px 18px", color: "var(--text-muted)" }}>Verdict</th>
                      <th style={{ padding: "14px 18px", color: "var(--text-muted)" }}>Confidence</th>
                      <th style={{ padding: "14px 18px", color: "var(--text-muted)" }}>Landmarks</th>
                      <th style={{ padding: "14px 18px", color: "var(--text-muted)" }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                        <td style={{ padding: "14px 18px", fontWeight: "500", fontFamily: "var(--font-mono)" }}>
                          {row.filename || `Scan #${row.id}`}
                        </td>
                        <td style={{ padding: "14px 18px" }}>
                          <span style={{
                            padding: "4px 10px",
                            borderRadius: "6px",
                            fontSize: "0.75rem",
                            fontWeight: "600",
                            background: row.prediction === "Real" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                            color: row.prediction === "Real" ? "#34d399" : "#f87171"
                          }}>
                            {row.prediction}
                          </span>
                        </td>
                        <td style={{ padding: "14px 18px", fontFamily: "var(--font-mono)" }}>
                          {row.confidence}%
                        </td>
                        <td style={{ padding: "14px 18px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                          {row.landmarks_count || 478} pts
                        </td>
                        <td style={{ padding: "14px 18px", color: "var(--text-dim)", fontSize: "0.8rem" }}>
                          {new Date(row.created_at || row.timestamp).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
                  No previous scans recorded yet. Run an analysis on the Scanner tab!
                </div>
              )}
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{
        borderTop: "1px solid var(--border-subtle)",
        padding: "16px 24px",
        textAlign: "center",
        fontSize: "0.8rem",
        color: "var(--text-dim)"
      }}>
        DeepTrace Mini Project • Explainable Facial Deepfake Detection Using Facial Landmark Geometry & Symmetry
      </footer>
    </div>
  );
}