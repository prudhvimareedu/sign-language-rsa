import { useEffect, useRef, useState } from "react";
import "./App.css";
import { predictImage } from "./services/api";

const SIGNS = ["A", "B", "C", "L", "O", "V", "Y"];

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const predictionHistoryRef = useRef([]);
  const lastSpokenRef = useRef("");
  const lastSpokenTimeRef = useRef(0);
  const speechEnabledRef = useRef(true);

  const [cameraActive, setCameraActive] = useState(false);
  const [prediction, setPrediction] = useState("—");
  const [handDetected, setHandDetected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [error, setError] = useState("");
  const [recentPredictions, setRecentPredictions] = useState([]);

  useEffect(() => {
    speechEnabledRef.current = speechEnabled;
  }, [speechEnabled]);

  const speakPrediction = (sign) => {
    if (!speechEnabledRef.current) return;

    if (
      !sign ||
      sign === "No hand" ||
      sign === "Unknown" ||
      sign === "—"
    ) {
      return;
    }

    if (!("speechSynthesis" in window)) return;

    const now = Date.now();
    const cooldownPassed =
      now - lastSpokenTimeRef.current > 2500;

    const changed = sign !== lastSpokenRef.current;

    if (!changed && !cooldownPassed) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(
      `Sign ${sign}`
    );

    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    window.speechSynthesis.speak(utterance);

    lastSpokenRef.current = sign;
    lastSpokenTimeRef.current = now;
  };

  const addPrediction = (value) => {
    const history = [
      ...predictionHistoryRef.current,
      value,
    ].slice(-5);

    predictionHistoryRef.current = history;

    const counts = {};

    history.forEach((item) => {
      counts[item] = (counts[item] || 0) + 1;
    });

    let best = history[history.length - 1];
    let bestCount = 0;

    Object.entries(counts).forEach(
      ([label, count]) => {
        if (count > bestCount) {
          best = label;
          bestCount = count;
        }
      }
    );

    setPrediction(best);

    setRecentPredictions((current) => {
      const next = [
        best,
        ...current.filter((item) => item !== best),
      ];

      return next.slice(0, 5);
    });

    speakPrediction(best);
  };

  const startCamera = async () => {
    try {
      setError("");

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: "user",
          },
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      predictionHistoryRef.current = [];
      lastSpokenRef.current = "";
      lastSpokenTimeRef.current = 0;

      setPrediction("—");
      setHandDetected(false);
      setRecentPredictions([]);
      setCameraActive(true);
    } catch (err) {
      console.error(err);
      setError(
        "Camera access was denied or unavailable."
      );
    }
  };

  const stopCamera = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    predictionHistoryRef.current = [];

    setCameraActive(false);
    setPrediction("—");
    setHandDetected(false);
    setRecentPredictions([]);
    setLoading(false);
  };

  const captureAndPredict = async () => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      videoRef.current.readyState < 2
    ) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) return;

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      async (blob) => {
        if (!blob) return;

        try {
          setLoading(true);

          const result = await predictImage(blob);

          if (result.hand_detected) {
            setHandDetected(true);

            addPrediction(
              result.prediction || "Unknown"
            );
          } else {
            setHandDetected(false);

            predictionHistoryRef.current = [];

            setPrediction("No hand");
          }

          setError("");
        } catch (err) {
          console.error(err);
          setError(
            "Prediction service unavailable."
          );
        } finally {
          setLoading(false);
        }
      },
      "image/jpeg",
      0.85
    );
  };

  useEffect(() => {
    if (!cameraActive) return;

    timerRef.current = setInterval(
      captureAndPredict,
      1000
    );

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [cameraActive]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">RSA</div>

          <div>
            <div className="brand-title">
              SignVision
            </div>

            <div className="brand-subtitle">
              AI Sign Language Recognition
            </div>
          </div>
        </div>

        <div className="topbar-right">
          <div className="system-status">
            <span className="pulse"></span>
            SYSTEM ONLINE
          </div>

          <div className="model-badge">
            KNN · LIVE
          </div>
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <div>
            <div className="eyebrow">
              REAL-TIME COMPUTER VISION
            </div>

            <h1>
              Sign language,
              <span> translated instantly.</span>
            </h1>

            <p>
              Webcam-based static ASL recognition
              powered by MediaPipe hand landmarks,
              normalized 63-feature representations
              and a KNN classifier.
            </p>
          </div>

          <div className="hero-chip">
            <span className="chip-dot"></span>
            7 LIVE CLASSES
          </div>
        </section>

        <section className="workspace">
          <div className="camera-panel glass-panel">
            <div className="panel-header">
              <div>
                <h2>Live Camera</h2>
                <p>
                  Position your hand inside the frame
                </p>
              </div>

              <div
                className={
                  cameraActive
                    ? "camera-state active"
                    : "camera-state"
                }
              >
                <span></span>

                {cameraActive
                  ? "ACTIVE"
                  : "OFFLINE"}
              </div>
            </div>

            <div className="video-shell">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-video"
              />

              {!cameraActive && (
                <div className="camera-empty">
                  <div className="camera-symbol">
                    ◉
                  </div>

                  <strong>
                    Camera not started
                  </strong>

                  <span>
                    Start your camera to begin
                  </span>
                </div>
              )}

              {cameraActive && (
                <>
                  <div className="scan-line"></div>

                  <div className="corner top-left"></div>
                  <div className="corner top-right"></div>
                  <div className="corner bottom-left"></div>
                  <div className="corner bottom-right"></div>
                </>
              )}
            </div>

            <canvas
              ref={canvasRef}
              style={{ display: "none" }}
            />

            <div className="camera-controls">
              {!cameraActive ? (
                <button
                  className="primary-btn"
                  onClick={startCamera}
                >
                  <span>▶</span>
                  Start Recognition
                </button>
              ) : (
                <button
                  className="secondary-btn"
                  onClick={stopCamera}
                >
                  <span>■</span>
                  Stop Camera
                </button>
              )}

              <button
                className={
                  speechEnabled
                    ? "voice-btn active"
                    : "voice-btn"
                }
                onClick={() =>
                  setSpeechEnabled(
                    (value) => !value
                  )
                }
              >
                {speechEnabled
                  ? "🔊 Voice On"
                  : "🔇 Voice Off"}
              </button>
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>

          <div className="side-column">
            <div className="prediction-panel glass-panel">
              <div className="panel-label">
                CURRENT PREDICTION
              </div>

              <div className="big-prediction">
                {loading ? "..." : prediction}
              </div>

              <div
                className={
                  handDetected
                    ? "detection active"
                    : "detection"
                }
              >
                <span></span>

                {handDetected
                  ? "Hand detected"
                  : "Waiting for hand"}
              </div>

              <div className="prediction-divider"></div>

              <div className="prediction-meta">
                <div>
                  <span>Live Pipeline</span>

                  <strong>
                    MediaPipe → Scaler → KNN
                  </strong>
                </div>

                <div>
                  <span>Live Features</span>

                  <strong>63 / 63</strong>
                </div>
              </div>
            </div>

            <div className="stats-grid">
              <div className="stat-card glass-panel">
                <span>
                  RSA Reduction
                </span>

                <strong>
                  44.44%
                </strong>

                <small>
                  63 → 35 research features
                </small>
              </div>

              <div className="stat-card glass-panel">
                <span>
                  Training Samples
                </span>

                <strong>
                  200
                </strong>

                <small>
                  7 calibrated sign classes
                </small>
              </div>
            </div>

            <div className="recent-panel glass-panel">
              <div className="panel-header compact">
                <div>
                  <h3>Recent Signs</h3>

                  <p>
                    Latest recognized classes
                  </p>
                </div>
              </div>

              {recentPredictions.length === 0 ? (
                <div className="empty-history">
                  No predictions yet
                </div>
              ) : (
                <div className="recent-list">
                  {recentPredictions.map(
                    (sign, index) => (
                      <div
                        className="recent-item"
                        key={`${sign}-${index}`}
                      >
                        <span className="recent-index">
                          0{index + 1}
                        </span>

                        <span className="recent-letter">
                          {sign}
                        </span>

                        <span className="recent-arrow">
                          →
                        </span>
                      </div>
                    )
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="sign-section glass-panel">
          <div className="sign-section-title">
            <div>
              <div className="panel-label">
                CALIBRATED CLASSES
              </div>

              <h2>Supported Signs</h2>
            </div>

            <span>7 classes</span>
          </div>

          <div className="sign-grid">
            {SIGNS.map((sign) => (
              <div
                key={sign}
                className={
                  prediction === sign
                    ? "sign-card selected"
                    : "sign-card"
                }
              >
                <div className="sign-letter">
                  {sign}
                </div>

                <div className="sign-name">
                  ASL {sign}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="pipeline-section">
          <div className="section-heading">
            <div className="eyebrow">
              SYSTEM PIPELINE
            </div>

            <h2>
              From webcam to prediction
            </h2>
          </div>

          <div className="pipeline-row">
            <div className="pipeline-card">
              <span>01</span>
              <strong>Webcam</strong>
              <small>
                Live image capture
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>02</span>
              <strong>MediaPipe</strong>
              <small>
                21 hand landmarks
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>03</span>
              <strong>Feature Scaling</strong>
              <small>
                63 normalized features
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>04</span>
              <strong>KNN</strong>
              <small>
                Sign classification
              </small>
            </div>
          </div>
        </section>

        <section className="pipeline-section">
          <div className="section-heading">
            <div className="eyebrow">
              RESEARCH COMPONENT
            </div>

            <h2>
              Reptile Search Algorithm
            </h2>
          </div>

          <div className="pipeline-row">
            <div className="pipeline-card">
              <span>01</span>
              <strong>63 Features</strong>
              <small>
                Full landmark representation
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>02</span>
              <strong>RSA</strong>
              <small>
                Feature selection
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>03</span>
              <strong>35 Features</strong>
              <small>
                Reduced feature space
              </small>
            </div>

            <div className="pipeline-connector">
              →
            </div>

            <div className="pipeline-card">
              <span>04</span>
              <strong>44.44%</strong>
              <small>
                Feature reduction
              </small>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>
          SignVision · ASL Recognition
        </span>

        <span>
          AI + Data Science
        </span>
      </footer>
    </div>
  );
}

export default App;