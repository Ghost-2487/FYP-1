import React, { useState } from "react";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from "chart.js";
import { depressionModels } from "../data/depressionData";
import { suicideModels }    from "../data/suicideData";

ChartJS.register(
  CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend, Filler
);

const PURPLE = "#7c71e8";
const CORAL  = "#e8714a";
const TEAL   = "#2dd4aa";
const AMBER  = "#f5a623";

function ConfusionMatrix({ model, isSui }) {
  const { tp, fp, fn, tn } = model.confusionMatrix;
  const cls = model.classes || (isSui ? ["Suicidal", "Non-Suicidal"] : ["Depressed", "Not Depressed"]);
  const accent = isSui ? CORAL : PURPLE;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "80px 1fr 1fr", gap: 4, marginBottom: 4 }}>
        <div />
        <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>Pred. Positive</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>Pred. Negative</div>
      </div>
      <div style={{ display: "grid", gap: 4 }}>
        <div style={{ display: "grid", gridTemplateColumns: "80px 1fr 1fr", gap: 4, alignItems: "center" }}>
          <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "right", paddingRight: 8 }}>Act. Positive</div>
          <div style={{ background: `${accent}33`, border: `1px solid ${accent}55`, borderRadius: 8, height: 70, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: accent, fontFamily: "var(--font-mono)" }}>{tp.toLocaleString()}%</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>True Positive</div>
          </div>
          <div style={{ background: "rgba(232,85,85,0.15)", border: "1px solid rgba(232,85,85,0.3)", borderRadius: 8, height: 70, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#e87777", fontFamily: "var(--font-mono)" }}>{fn.toLocaleString()}%</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>False Negative</div>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "80px 1fr 1fr", gap: 4, alignItems: "center" }}>
          <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "right", paddingRight: 8 }}>Act. Negative</div>
          <div style={{ background: "rgba(232,85,85,0.15)", border: "1px solid rgba(232,85,85,0.3)", borderRadius: 8, height: 70, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#e87777", fontFamily: "var(--font-mono)" }}>{fp.toLocaleString()}%</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>False Positive</div>
          </div>
          <div style={{ background: `${accent}33`, border: `1px solid ${accent}55`, borderRadius: 8, height: 70, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: accent, fontFamily: "var(--font-mono)" }}>{tn.toLocaleString()}%</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>True Negative</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RocCurve({ model, isSui }) {
  const accent = isSui ? CORAL : PURPLE;
  const pts = model.rocCurve;
  const rocData = {
    datasets: [
      {
        label: `ROC (AUC = ${model.aucRoc.toFixed(3)})`,
        data: pts.map(([x, y]) => ({ x, y })),
        borderColor: accent,
        backgroundColor: accent + "22",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      },
      {
        label: "Random classifier",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(255,255,255,0.2)",
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
        borderWidth: 1,
      },
    ],
  };
  const rocOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` FPR: ${ctx.raw.x.toFixed(2)}, TPR: ${ctx.raw.y.toFixed(2)}` } },
    },
    scales: {
      x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate", color: "#9999aa", font: { size: 11 } }, ticks: { color: "#9999aa", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
      y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True Positive Rate", color: "#9999aa", font: { size: 11 } }, ticks: { color: "#9999aa", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  };
  return <Line data={rocData} options={rocOpts} />;
}

function PerClassMetrics({ model, isSui }) {
  const accent = isSui ? CORAL : PURPLE;
  const classes = Object.keys(model.classMetrics);
  const metrics = ["precision", "recall", "f1"];
  const colors  = [accent, TEAL, AMBER];
  const data = {
    labels: classes,
    datasets: metrics.map((m, i) => ({
      label: m.charAt(0).toUpperCase() + m.slice(1) + " (%)",
      data: classes.map((c) => model.classMetrics[c][m]),
      backgroundColor: colors[i] + "bb",
      borderRadius: 4,
      borderSkipped: false,
    })),
  };
  const opts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } },
    },
    scales: {
      x: { ticks: { color: "#9999aa" }, grid: { display: false } },
      y: { min: 60, max: 100, ticks: { color: "#9999aa", callback: (v) => v + "%" }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  };
  return <Bar data={data} options={opts} />;
}

// ——— MODEL DETAIL PAGE ———
function ModelDetail({ model, isSui, onBack }) {
  const accent = isSui ? CORAL : PURPLE;
  const isDL   = model.type === "DL";

  return (
    <div>
      <button className="back-btn" onClick={onBack}>← Back to all models</button>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
        <h2 style={{ fontSize: 22, fontWeight: 600 }}>{model.name}</h2>
        <span className={`badge ${model.type === "DL" ? "dl" : "ml"}`}>{model.category}</span>
        <span className={`badge ${isSui ? "sui" : "dep"}`}>{isSui ? "Suicide" : "Depression"}</span>
      </div>

      {/* Metrics row */}
      <div className="metric-grid" style={{ marginBottom: 20 }}>
        {[
          { label: "Accuracy",  val: model.accuracy.toFixed(1) + "%",  color: accent },
          { label: "Precision", val: model.precision.toFixed(1) + "%", color: accent },
          { label: "Recall",    val: model.recall.toFixed(1) + "%",    color: TEAL },
          { label: "F1-score",  val: model.f1.toFixed(1) + "%",        color: TEAL },
          { label: "AUC-ROC",   val: model.aucRoc.toFixed(3),          color: AMBER },
          { label: "Vectorization", val: model.vectorization, color: "var(--text-secondary)" },
          { label: "Train time",
            val: model.trainingTimeMinutes >= 60
              ? `~${(model.trainingTimeMinutes / 60).toFixed(1)}h`
              : model.trainingTimeMinutes < 2 ? "<1 min" : `~${model.trainingTimeMinutes} min`,
            color: "var(--text-secondary)" },
          { label: "Epochs", val: model.epochs ? model.epochs : "N/A", color: "var(--text-secondary)" },
        ].map((m) => (
          <div className="metric-card" key={m.label}>
            <div className="metric-label">{m.label}</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: "var(--font-mono)", color: m.color, marginTop: 4 }}>{m.val}</div>
          </div>
        ))}
      </div>

      {/* Notes */}
      {model.notes && (
        <div className="card" style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)", marginRight: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>Notes</span>
            {model.notes}
          </div>
        </div>
      )}

      <div className="grid2">
        {/* Confusion matrix */}
        <div className="card">
          <div className="card-title">Confusion matrix</div>
          <ConfusionMatrix model={model} isSui={isSui} />
        </div>
        {/* ROC curve */}
        <div className="card">
          <div className="card-title">ROC curve</div>
          <div style={{ height: 220 }}>
            <RocCurve model={model} isSui={isSui} />
          </div>
        </div>
      </div>

      {/* Per-class metrics */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">Per-class metrics</div>
        <div style={{ height: 200 }}>
          <PerClassMetrics model={model} isSui={isSui} />
        </div>
      </div>
    </div>
  );
}

// ——— MODEL LIST VIEW ———
export default function ModelResults({ activeDataset }) {
  const [selectedModel, setSelectedModel] = useState(null);
  const isSui  = activeDataset === "suicide";
  const models = isSui ? suicideModels : depressionModels;
  const accent = isSui ? CORAL : PURPLE;

  const sorted = [...models].sort((a, b) => b.accuracy - a.accuracy);

  if (selectedModel) {
    return (
      <ModelDetail
        model={selectedModel}
        isSui={isSui}
        onBack={() => setSelectedModel(null)}
      />
    );
  }

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <div className="section-label">Select a model to view detailed results</div>
        <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
          Showing {isSui ? "Suicide" : "Depression"} dataset · {models.length} models · sorted by accuracy
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 20, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: accent }} />
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Deep Learning</span>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: TEAL }} />
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Classical ML</span>
        </div>
      </div>

      <div className="model-list">
        {sorted.map((model, idx) => (
          <button
            key={model.id}
            className="model-row"
            onClick={() => setSelectedModel(model)}
          >
            <div className="model-row-left">
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: model.type === "DL" ? accent + "22" : TEAL + "22",
                border: `1px solid ${model.type === "DL" ? accent + "44" : TEAL + "44"}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 12, fontWeight: 700, color: model.type === "DL" ? accent : TEAL,
                flexShrink: 0,
              }}>
                {idx + 1}
              </div>
              <div>
                <div className="model-row-name">{model.name}</div>
                <div className="model-row-type">
                  {model.category} · {model.vectorization}
                  {model.epochs ? ` · ${model.epochs} epochs` : ""}
                </div>
              </div>
            </div>
            <div className="model-row-stats">
              <div className="model-stat">
                <div className="model-stat-val" style={{ color: accent }}>{model.accuracy.toFixed(1)}%</div>
                <div className="model-stat-key">Accuracy</div>
              </div>
              <div className="model-stat">
                <div className="model-stat-val" style={{ color: TEAL }}>{model.f1.toFixed(1)}%</div>
                <div className="model-stat-key">F1-score</div>
              </div>
              <div className="model-stat">
                <div className="model-stat-val" style={{ color: "var(--text-secondary)" }}>{model.aucRoc.toFixed(3)}</div>
                <div className="model-stat-key">AUC-ROC</div>
              </div>
              <div className="model-stat">
                <div className="model-stat-val" style={{ color: "var(--text-secondary)", fontSize: 13 }}>
                  {model.trainingTimeMinutes >= 60
                    ? `~${(model.trainingTimeMinutes / 60).toFixed(1)}h`
                    : model.trainingTimeMinutes < 2 ? "<1m" : `~${model.trainingTimeMinutes}m`}
                </div>
                <div className="model-stat-key">Train time</div>
              </div>
            </div>
            <span className="model-row-arrow">›</span>
          </button>
        ))}
      </div>
    </div>
  );
}
