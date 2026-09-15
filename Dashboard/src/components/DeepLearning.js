import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from "chart.js";
import { depressionModels } from "../data/depressionData";
import { suicideModels }    from "../data/suicideData";

ChartJS.register(
  CategoryScale, LinearScale,
  PointElement, LineElement, Tooltip, Legend, Filler
);

const PURPLE = "#7c71e8";
const CORAL  = "#e8714a";
const TEAL   = "#2dd4aa";
const AMBER  = "#f5a623";

const DL_IDS   = ["cnn", "lstm", "bilstm", "bert"];
const DL_NAMES = { cnn: "CNN", lstm: "LSTM", bilstm: "BiLSTM", bert: "BERT" };
const DL_COLORS= { cnn: AMBER, lstm: TEAL, bilstm: PURPLE, bert: CORAL };

function TrainingCurve({ model, isSui }) {
  const [view, setView] = useState("loss");
  const accent = DL_COLORS[model.id] || PURPLE;
  const h = model.trainingHistory;
  if (!h) return <p style={{ color: "var(--text-muted)", fontSize: 12 }}>No training history available.</p>;

  const epochs = Array.from({ length: h.trainLoss.length }, (_, i) => `Epoch ${i + 1}`);
  const isLoss = view === "loss";

  const chartData = {
    labels: epochs,
    datasets: [
      {
        label: isLoss ? "Train loss" : "Train accuracy",
        data: isLoss ? h.trainLoss : h.trainAccuracy.map((v) => v * 100),
        borderColor: accent,
        backgroundColor: accent + "22",
        fill: false,
        tension: 0.3,
        pointRadius: epochs.length <= 5 ? 4 : 2,
        borderWidth: 2,
      },
      {
        label: isLoss ? "Val loss" : "Val accuracy",
        data: isLoss ? h.valLoss : h.valAccuracy.map((v) => v * 100),
        borderColor: accent + "88",
        borderDash: [5, 4],
        backgroundColor: "transparent",
        fill: false,
        tension: 0.3,
        pointRadius: epochs.length <= 5 ? 4 : 2,
        borderWidth: 2,
      },
    ],
  };

  const opts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(isLoss ? 3 : 1)}${isLoss ? "" : "%"}` } },
    },
    scales: {
      x: {
        ticks: {
          color: "#9999aa", font: { size: 10 },
          autoSkip: epochs.length > 10,
          maxTicksLimit: 10,
        },
        grid: { color: "rgba(255,255,255,0.04)" },
      },
      y: {
        ticks: { color: "#9999aa", callback: (v) => isLoss ? v.toFixed(2) : v.toFixed(0) + "%" },
        grid: { color: "rgba(255,255,255,0.06)" },
        ...(isLoss ? {} : { min: 50, max: 100 }),
      },
    },
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
        {["loss", "accuracy"].map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              font: "500 11px var(--font-sans)",
              padding: "4px 12px",
              borderRadius: 100,
              border: `1px solid ${view === v ? accent : "var(--border)"}`,
              background: view === v ? accent + "22" : "transparent",
              color: view === v ? accent : "var(--text-muted)",
              cursor: "pointer",
            }}
          >
            {v.charAt(0).toUpperCase() + v.slice(1)}
          </button>
        ))}
      </div>
      <div style={{ height: 180 }}>
        <Line data={chartData} options={opts} />
      </div>
    </div>
  );
}

function MetricBar({ label, val, max = 100, color }) {
  return (
    <div className="bar-row">
      <div className="bar-label" style={{ width: 80 }}>{label}</div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: (val / max * 100) + "%", background: color }} />
      </div>
      <div className="bar-val">{val.toFixed(1)}{max === 100 ? "%" : ""}</div>
    </div>
  );
}

export default function DeepLearning({ activeDataset }) {
  const [selected, setSelected] = useState(null);
  const isSui  = activeDataset === "suicide";
  const models = (isSui ? suicideModels : depressionModels).filter((m) => DL_IDS.includes(m.id));
  const accent = isSui ? CORAL : PURPLE;

  const sorted = DL_IDS.map((id) => models.find((m) => m.id === id)).filter(Boolean);

  // All-model overlay chart
  const epochs20 = Array.from({ length: 8 }, (_, i) => `${i + 1}`);
  const overlayData = (key) => ({
    labels: epochs20,
    datasets: sorted
      .filter((m) => m.trainingHistory && m.id !== "bert")
      .map((m) => ({
        label: m.shortName,
        data: m.trainingHistory[key],
        borderColor: DL_COLORS[m.id],
        backgroundColor: "transparent",
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      })),
  });

  const overlayOpts = (isLoss) => ({
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(isLoss ? 3 : 2)}` } },
    },
    scales: {
      x: { ticks: { color: "#9999aa", font: { size: 10 }, maxTicksLimit: 20 }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { ticks: { color: "#9999aa", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  });

  if (selected) {
    const model = models.find((m) => m.id === selected);
    const c = DL_COLORS[model.id];
    return (
      <div>
        <button className="back-btn" onClick={() => setSelected(null)}>← Back to all DL models</button>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600 }}>{model.name}</h2>
          <span className={`badge ${isSui ? "sui" : "dep"}`}>{isSui ? "Suicide" : "Depression"}</span>
        </div>

        <div className="metric-grid" style={{ marginBottom: 16 }}>
          {[
            { label: "Accuracy",  val: model.accuracy.toFixed(1) + "%",  col: c },
            { label: "F1-score",  val: model.f1.toFixed(1) + "%",        col: TEAL },
            { label: "Precision", val: model.precision.toFixed(1) + "%", col: AMBER },
            { label: "Recall",    val: model.recall.toFixed(1) + "%",    col: AMBER },
            { label: "AUC-ROC",   val: model.aucRoc.toFixed(3),          col: c },
            { label: "Epochs",    val: model.epochs || "—",              col: "var(--text-secondary)" },
            { label: "Embedding", val: model.vectorization,              col: "var(--text-secondary)" },
            { label: "Train time",
              val: `~${(model.trainingTimeMinutes / 60).toFixed(1)}h`,
              col: "var(--text-secondary)" },
          ].map((item) => (
            <div className="metric-card" key={item.label}>
              <div className="metric-label">{item.label}</div>
              <div style={{ fontSize: 17, fontWeight: 600, fontFamily: "var(--font-mono)", color: item.col, marginTop: 4 }}>{item.val}</div>
            </div>
          ))}
        </div>

        <div className="card" style={{ marginBottom: 14 }}>
          <div className="card-title">Training curves — {model.name}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
            Solid = training · Dashed = validation
          </div>
          <TrainingCurve model={model} isSui={isSui} />
        </div>

        <div className="card">
          <div className="card-title">Performance breakdown</div>
          <MetricBar label="Accuracy"  val={model.accuracy}  color={c} />
          <MetricBar label="Precision" val={model.precision} color={c} />
          <MetricBar label="Recall"    val={model.recall}    color={TEAL} />
          <MetricBar label="F1-score"  val={model.f1}        color={TEAL} />
          <div style={{ marginTop: 10 }}>
            <MetricBar label="AUC-ROC" val={model.aucRoc * 100} color={AMBER} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="section-label">Deep learning models — {isSui ? "Suicide" : "Depression"} dataset</div>

      {/* Model cards */}
      <div className="grid4" style={{ marginBottom: 20 }}>
        {sorted.map((model) => {
          const c = DL_COLORS[model.id];
          return (
            <button
              key={model.id}
              onClick={() => setSelected(model.id)}
              style={{
                background: "var(--bg-card)",
                border: `1px solid var(--border)`,
                borderRadius: "var(--radius-lg)",
                padding: "14px 16px",
                cursor: "pointer",
                textAlign: "left",
                font: "var(--font-sans)",
                transition: "border-color 0.15s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = c + "88"}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = "var(--border)"}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>{DL_NAMES[model.id]}</div>
                <div style={{ fontSize: 12, padding: "2px 8px", borderRadius: 100, background: c + "22", color: c, border: `1px solid ${c}44` }}>
                  {model.category}
                </div>
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: c, fontFamily: "var(--font-mono)", marginBottom: 4 }}>
                {model.accuracy.toFixed(1)}%
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>accuracy</div>
              <div style={{ marginTop: 10, display: "flex", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "var(--font-mono)", color: TEAL }}>{model.f1.toFixed(1)}%</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>F1</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{model.aucRoc.toFixed(3)}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>AUC</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    {model.epochs ? `${model.epochs}ep` : "—"}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>epochs</div>
                </div>
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: c + "99" }}>Click to view details →</div>
            </button>
          );
        })}
      </div>

      {/* Overlay: all models loss curves */}
      <div className="grid2">
        <div className="card">
          <div className="card-title">Validation loss — all DL models (except BERT)</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
            BERT excluded (only 3 epochs). Lower = better.
          </div>
          <div style={{ height: 220 }}>
            <Line data={overlayData("valLoss")} options={overlayOpts(true)} />
          </div>
        </div>
        <div className="card">
          <div className="card-title">Validation accuracy — all DL models (except BERT)</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
            Expressed as ratio (0–1). Higher = better.
          </div>
          <div style={{ height: 220 }}>
            <Line data={overlayData("valAccuracy")} options={overlayOpts(false)} />
          </div>
        </div>
      </div>

      {/* BERT special section */}
      {(() => {
        const bert = models.find((m) => m.id === "bert");
        if (!bert) return null;
        const c = DL_COLORS.bert;
        const epochs3 = ["Epoch 1", "Epoch 2", "Epoch 3"];
        const bertData = {
          labels: epochs3,
          datasets: [
            { label: "Train loss", data: bert.trainingHistory.trainLoss, borderColor: c, fill: false, tension: 0.3, pointRadius: 5, borderWidth: 2 },
            { label: "Val loss",   data: bert.trainingHistory.valLoss,   borderColor: c + "77", borderDash: [5,4], fill: false, tension: 0.3, pointRadius: 5, borderWidth: 2 },
          ],
        };
        const bertAccData = {
          labels: epochs3,
          datasets: [
            { label: "Train acc", data: bert.trainingHistory.trainAccuracy.map((v) => v * 100), borderColor: c, fill: false, tension: 0.3, pointRadius: 5, borderWidth: 2 },
            { label: "Val acc",   data: bert.trainingHistory.valAccuracy.map((v) => v * 100),   borderColor: c + "77", borderDash: [5,4], fill: false, tension: 0.3, pointRadius: 5, borderWidth: 2 },
          ],
        };
        const bertOpts = (isLoss) => ({
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
            tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(isLoss ? 3 : 1)}${isLoss ? "" : "%"}` } },
          },
          scales: {
            x: { ticks: { color: "#9999aa" }, grid: { color: "rgba(255,255,255,0.04)" } },
            y: { ticks: { color: "#9999aa" }, grid: { color: "rgba(255,255,255,0.06)" }, ...(isLoss ? {} : { min: 70, max: 100 }) },
          },
        });
        return (
          <div className="card" style={{ marginTop: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <div className="card-title" style={{ margin: 0 }}>BERT fine-tuning (3 epochs)</div>
              <span className="badge dep" style={{ background: c + "22", color: c, borderColor: c + "44" }}>bert-base-uncased</span>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 14 }}>
              BERT is fine-tuned for only 3 epochs — typical for transformer models to avoid overfitting.
              Max sequence length: 128 · Batch size: 32
            </div>
            <div className="grid2">
              <div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8 }}>Loss per epoch</div>
                <div style={{ height: 180 }}><Line data={bertData} options={bertOpts(true)} /></div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8 }}>Accuracy per epoch (%)</div>
                <div style={{ height: 180 }}><Line data={bertAccData} options={bertOpts(false)} /></div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
