import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Tooltip, Legend,
} from "chart.js";
import { depressionModels, depressionDataset } from "../data/depressionData";
import { suicideModels, suicideDataset } from "../data/suicideData";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const PURPLE = "#7c71e8";
const CORAL  = "#e8714a";
const TEAL   = "#2dd4aa";
const ML_COLOR = "#2dd4aa";

function ConfusionMatrix({ model, dataset }) {
  const isSui = dataset === "suicide";
  const { tp, fp, fn, tn } = model.confusionMatrix;
  const cls = isSui ? "sui" : "";
  return (
    <div className="cm-wrapper">
      <div className="cm-axis-labels" style={{ gridTemplateColumns: "24px 1fr 1fr" }}>
        <div />
        <div className="cm-col-label">Pred. Positive</div>
        <div className="cm-col-label">Pred. Negative</div>
      </div>
      <div className="cm-rows" style={{ gap: 2 }}>
        <div className="cm-row">
          <div className="cm-row-label" style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>Act+</div>
          <div className={`cm-cell tp ${cls}`}>{tp.toLocaleString()}%</div>
          <div className="cm-cell fn">{fn.toLocaleString()}%</div>
        </div>
        <div className="cm-row">
          <div className="cm-row-label" style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>Act-</div>
          <div className="cm-cell fp">{fp.toLocaleString()}%</div>
          <div className={`cm-cell tn ${cls}`}>{tn.toLocaleString()}%</div>
        </div>
      </div>
    </div>
  );
}

export default function Overview({ activeDataset }) {
  const isSui = activeDataset === "suicide";
  const models = isSui ? suicideModels : depressionModels;
  const dataset = isSui ? suicideDataset : depressionDataset;
  const accent = isSui ? CORAL : PURPLE;

  const bestAcc = [...models].sort((a, b) => b.accuracy - a.accuracy)[0];
  const bestF1  = [...models].sort((a, b) => b.f1 - a.f1)[0];
  const bestAuc = [...models].sort((a, b) => b.aucRoc - a.aucRoc)[0];
  const bestRecall = [...models].sort((a, b) => b.recall - a.recall)[0];

  const sorted = [...models].sort((a, b) => b.accuracy - a.accuracy);

  const barData = {
    labels: sorted.map((m) => m.shortName),
    datasets: [
      {
        label: "Accuracy (%)",
        data: sorted.map((m) => m.accuracy),
        backgroundColor: sorted.map((m) =>
          m.type === "DL" ? accent : ML_COLOR
        ),
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.raw.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#9999aa", font: { size: 11, family: "IBM Plex Sans" } },
        grid: { color: "rgba(255,255,255,0.04)" },
      },
      y: {
        min: 60,
        max: 100,
        ticks: {
          color: "#9999aa",
          font: { size: 11 },
          callback: (v) => v + "%",
        },
        grid: { color: "rgba(255,255,255,0.06)" },
      },
    },
  };

  const trainData = {
    labels: sorted.map((m) => m.shortName),
    datasets: [
      {
        label: "Training time (min)",
        data: sorted.map((m) => m.trainingTimeMinutes),
        backgroundColor: sorted.map((m) =>
          m.type === "DL" ? "rgba(232,113,74,0.7)" : "rgba(45,212,170,0.7)"
        ),
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const trainOptions = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw} min` } } },
    scales: {
      x: {
        ticks: { color: "#9999aa", font: { size: 11 }, stepSize: 30, callback: (v) => (v >= 60 ? (v / 60) + "h" : v + "m") },
        grid:  { color: "rgba(255,255,255,0.06)" },
      },
      y: { ticks: { color: "#9999aa", font: { size: 11 } }, grid: { display: false } },
    },
  };

  return (
    <div>
      {/* Summary metrics */}
      <div className="section-label">Summary — best performing models</div>
      <div className="metric-grid" style={{ marginBottom: 20 }}>
        <div className="metric-card">
          <div className="metric-label">Best accuracy</div>
          <div className="metric-value" style={{ color: accent }}>{bestAcc.accuracy.toFixed(1)}%</div>
          <div className="metric-sub">{bestAcc.name}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Best F1-score</div>
          <div className="metric-value" style={{ color: accent }}>{bestF1.f1.toFixed(1)}%</div>
          <div className="metric-sub">{bestF1.name}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Best AUC-ROC</div>
          <div className="metric-value" style={{ color: TEAL }}>{bestAuc.aucRoc.toFixed(3)}</div>
          <div className="metric-sub">{bestAuc.name}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Best recall</div>
          <div className="metric-value" style={{ color: TEAL }}>{bestRecall.recall.toFixed(1)}%</div>
          <div className="metric-sub">{bestRecall.name}</div>
        </div>
      </div>

      {/* Dataset badge + bar charts */}
      <div style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
        <span className={`badge ${isSui ? "sui" : "dep"}`}>
          {dataset.name}
        </span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
          {dataset.totalSamples.toLocaleString()} samples · {dataset.platform}
        </span>
      </div>

      <div className="grid2">
        <div className="card">
          <div className="card-title">Accuracy — all models</div>
          <div className="legend-row">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: accent }} />
              DL models
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: ML_COLOR }} />
              ML models
            </div>
          </div>
          <div className="chart-wrapper" style={{ height: 220 }}>
            <Bar data={barData} options={barOptions} />
          </div>
        </div>

        <div className="card">
          <div className="card-title">Training time comparison</div>
          <div className="legend-row">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: "rgba(232,113,74,0.7)" }} />
              DL models
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: "rgba(45,212,170,0.7)" }} />
              ML models
            </div>
          </div>
          <div className="chart-wrapper" style={{ height: 220 }}>
            <Bar data={trainData} options={trainOptions} />
          </div>
        </div>
      </div>

      {/* Top model confusion matrix + dataset overview */}
      <div className="grid2">
        <div className="card">
          <div className="card-title">
            Best model confusion matrix &nbsp;
            <span className={`badge ${isSui ? "sui" : "dep"}`}>{bestAcc.shortName}</span>
          </div>
          <ConfusionMatrix model={bestAcc} dataset={activeDataset} />
          <div style={{ display: "flex", gap: 20, marginTop: 14, justifyContent: "center" }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Precision</div>
              <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 16 }}>{bestAcc.precision.toFixed(1)}%</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Recall</div>
              <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 16 }}>{bestAcc.recall.toFixed(1)}%</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>F1-score</div>
              <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 16 }}>{bestAcc.f1.toFixed(1)}%</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>AUC-ROC</div>
              <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 16 }}>{bestAcc.aucRoc.toFixed(3)}</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Dataset overview</div>
          <div className="info-row"><span className="key">Total samples</span><span className="val">{dataset.totalSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Training set</span><span className="val">{dataset.trainSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Validation set</span><span className="val">{dataset.valSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Test set</span><span className="val">{dataset.testSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Avg text length</span><span className="val">~{dataset.avgTextLength} tokens</span></div>
          <div className="info-row"><span className="key">Vocabulary size</span><span className="val">{dataset.vocabularySize.toLocaleString()}</span></div>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>Class distribution</div>
            <div className="class-bar">
              {Object.entries(dataset.classDistribution).map(([cls, pct], i) => (
                <div
                  key={cls}
                  className="class-bar-seg"
                  style={{
                    width: pct + "%",
                    background: i === 0 ? accent : "rgba(255,255,255,0.12)",
                  }}
                />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
              {Object.entries(dataset.classDistribution).map(([cls, pct]) => (
                <span key={cls}>{cls} {pct}%</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Full comparison table */}
      <div className="card">
        <div className="card-title">Full model comparison — {dataset.name}</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-score</th>
              <th>AUC-ROC</th>
              <th>Train time</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => (
              <tr key={m.id}>
                <td>{m.name}</td>
                <td>
                  <span className={`badge ${m.type === "DL" ? "dl" : "ml"}`}>{m.category}</span>
                </td>
                <td style={{ fontFamily: "var(--font-mono)" }}>
                  {m.id === bestAcc.id
                    ? <span className={`badge ${isSui ? "best-coral" : "best"}`}>{m.accuracy.toFixed(1)}%</span>
                    : `${m.accuracy.toFixed(1)}%`}
                </td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{m.precision.toFixed(1)}%</td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{m.recall.toFixed(1)}%</td>
                <td style={{ fontFamily: "var(--font-mono)" }}>
                  {m.id === bestF1.id
                    ? <span className={`badge ${isSui ? "best-coral" : "best"}`}>{m.f1.toFixed(1)}%</span>
                    : `${m.f1.toFixed(1)}%`}
                </td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{m.aucRoc.toFixed(3)}</td>
                <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  {m.trainingTimeMinutes >= 60
                    ? `~${(m.trainingTimeMinutes / 60).toFixed(1)}h`
                    : m.trainingTimeMinutes < 2 ? "<1 min" : `~${m.trainingTimeMinutes} min`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
