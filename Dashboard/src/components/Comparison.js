import React, { useState } from "react";
import { Bar, Radar, Scatter, Line } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  RadialLinearScale, PointElement, LineElement,
  Tooltip, Legend, Filler,
} from "chart.js";
import { depressionModels } from "../data/depressionData";
import { suicideModels }    from "../data/suicideData";

ChartJS.register(
  CategoryScale, LinearScale, BarElement,
  RadialLinearScale, PointElement, LineElement,
  Tooltip, Legend, Filler
);

const PURPLE = "#7c71e8";
const CORAL  = "#e8714a";
const TEAL   = "#2dd4aa";
const AMBER  = "#f5a623";
const GREEN  = "#4caf82";
const RED    = "#e85555";

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1"];
const METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-score"];

const MODEL_COLORS = {
  logistic_regression: "#9999cc",
  naive_bayes:         "#cc99bb",
  random_forest:       TEAL,
  svm:                 "#4dc9f6",
  cnn:                 AMBER,
  lstm:                "#f67019",
  bilstm:              "#537bc4",
  bert:                PURPLE,
};

export default function Comparison() {
  const [metric, setMetric] = useState("accuracy");

  const depModels = depressionModels;
  const suiModels = suicideModels;
  const allNames  = depModels.map((m) => m.shortName);

  // ── Grouped bar: both datasets side by side ──
  const groupedData = {
    labels: allNames,
    datasets: [
      {
        label: "Depression",
        data: depModels.map((m) => m[metric]),
        backgroundColor: PURPLE + "cc",
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: "Suicide",
        data: suiModels.map((m) => {
          const match = suiModels.find((s) => s.id === depModels[depModels.findIndex((d) => d.shortName === m.shortName)].id);
          return match ? match[metric] : 0;
        }),
        backgroundColor: CORAL + "cc",
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const groupedOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } },
    },
    scales: {
      x: { ticks: { color: "#9999aa", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: {
        min: 60, max: 100,
        ticks: { color: "#9999aa", callback: (v) => v + "%" },
        grid: { color: "rgba(255,255,255,0.06)" },
      },
    },
  };

  // ── Radar: all metrics for top models ──
  const TOP_IDS = ["bert", "bilstm", "lstm", "svm"];
  const radarLabels = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC (×100)"];
  const radarColors = [PURPLE, TEAL, AMBER, "#4dc9f6"];

  const radarData = {
    labels: radarLabels,
    datasets: TOP_IDS.map((id, i) => {
      const m = depModels.find((x) => x.id === id);
      return {
        label: m.shortName,
        data: [m.accuracy, m.precision, m.recall, m.f1, m.aucRoc * 100],
        borderColor: radarColors[i],
        backgroundColor: radarColors[i] + "22",
        pointBackgroundColor: radarColors[i],
        borderWidth: 2,
        pointRadius: 3,
      };
    }),
  };

  const radarOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } } },
    scales: {
      r: {
        min: 60, max: 100,
        ticks: { color: "#9999aa", font: { size: 9 }, backdropColor: "transparent", stepSize: 10 },
        grid:  { color: "rgba(255,255,255,0.08)" },
        pointLabels: { color: "#9999aa", font: { size: 11 } },
        angleLines:  { color: "rgba(255,255,255,0.08)" },
      },
    },
  };

  // ── Scatter: accuracy vs training time ──
  const allForScatter = depModels.map((m) => ({
    x: m.trainingTimeMinutes,
    y: m.accuracy,
    label: m.shortName,
    type: m.type,
    id: m.id,
  }));

  const scatterData = {
    datasets: [
      {
        label: "ML models",
        data: allForScatter.filter((p) => p.type === "ML").map((p) => ({ x: p.x, y: p.y, label: p.label })),
        backgroundColor: TEAL + "cc",
        pointRadius: 8,
        pointHoverRadius: 10,
      },
      {
        label: "DL models",
        data: allForScatter.filter((p) => p.type === "DL").map((p) => ({ x: p.x, y: p.y, label: p.label })),
        backgroundColor: PURPLE + "cc",
        pointRadius: 8,
        pointHoverRadius: 10,
      },
    ],
  };

  const scatterOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const pt = ctx.raw;
            return ` ${pt.label}: ${pt.y.toFixed(1)}% acc · ${pt.x >= 60 ? (pt.x / 60).toFixed(1) + "h" : pt.x + " min"}`;
          },
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Training time (minutes)", color: "#9999aa", font: { size: 11 } },
        ticks: { color: "#9999aa", callback: (v) => v >= 60 ? (v / 60).toFixed(0) + "h" : v + "m" },
        grid:  { color: "rgba(255,255,255,0.06)" },
      },
      y: {
        min: 70, max: 100,
        title: { display: true, text: "Accuracy (%)", color: "#9999aa", font: { size: 11 } },
        ticks: { color: "#9999aa", callback: (v) => v + "%" },
        grid:  { color: "rgba(255,255,255,0.06)" },
      },
    },
  };

  // ── ML vs DL group averages ──
  const avgMetric = (arr, key) => (arr.reduce((s, m) => s + m[key], 0) / arr.length);

  const depML = depModels.filter((m) => m.type === "ML");
  const depDL = depModels.filter((m) => m.type === "DL");
  const suiML = suiModels.filter((m) => m.type === "ML");
  const suiDL = suiModels.filter((m) => m.type === "DL");

  const groupAvgData = {
    labels: METRIC_LABELS,
    datasets: [
      {
        label: "ML (Depression)",
        data: METRIC_KEYS.map((k) => avgMetric(depML, k)),
        backgroundColor: TEAL + "99",
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: "DL (Depression)",
        data: METRIC_KEYS.map((k) => avgMetric(depDL, k)),
        backgroundColor: PURPLE + "99",
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: "ML (Suicide)",
        data: METRIC_KEYS.map((k) => avgMetric(suiML, k)),
        backgroundColor: TEAL + "55",
        borderColor: TEAL,
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: "DL (Suicide)",
        data: METRIC_KEYS.map((k) => avgMetric(suiDL, k)),
        backgroundColor: CORAL + "55",
        borderColor: CORAL,
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const groupAvgOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#9999aa", font: { size: 11 }, boxWidth: 12 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } },
    },
    scales: {
      x: { ticks: { color: "#9999aa" }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { min: 70, max: 100, ticks: { color: "#9999aa", callback: (v) => v + "%" }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  };

  return (
    <div>
      <div className="section-label">Cross-dataset model comparison</div>

      {/* Metric selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {METRIC_KEYS.map((k, i) => (
          <button
            key={k}
            onClick={() => setMetric(k)}
            style={{
              font: "500 12px var(--font-sans)",
              padding: "6px 14px",
              borderRadius: 100,
              border: `1px solid ${metric === k ? PURPLE : "var(--border-strong)"}`,
              background: metric === k ? "rgba(124,113,232,0.15)" : "transparent",
              color: metric === k ? "#a89ef5" : "var(--text-secondary)",
              cursor: "pointer",
            }}
          >
            {METRIC_LABELS[i]}
          </button>
        ))}
      </div>

      {/* Side-by-side both datasets */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">
          {METRIC_LABELS[METRIC_KEYS.indexOf(metric)]} — Depression vs Suicide datasets
        </div>
        <div className="legend-row">
          <div className="legend-item"><div className="legend-dot" style={{ background: PURPLE }} />Depression dataset</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: CORAL }} />Suicide dataset</div>
        </div>
        <div className="chart-wrapper" style={{ height: 260 }}>
          <Bar data={groupedData} options={groupedOpts} />
        </div>
      </div>

      <div className="grid2">
        {/* Radar chart */}
        <div className="card">
          <div className="card-title">Radar — top 4 models (Depression dataset)</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
            All metrics normalized to percentage scale
          </div>
          <div style={{ height: 280, position: "relative" }}>
            <Radar data={radarData} options={radarOpts} />
          </div>
        </div>

        {/* Scatter */}
        <div className="card">
          <div className="card-title">Accuracy vs training time (Depression)</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
            Top-right = best trade-off: high accuracy, low cost
          </div>
          <div style={{ height: 280, position: "relative" }}>
            <Scatter data={scatterData} options={scatterOpts} />
          </div>
        </div>
      </div>

      {/* ML vs DL group comparison */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">ML vs Deep Learning — average performance across all metrics</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
          Averaged across all 4 ML models and all 4 DL models per dataset
        </div>
        <div className="chart-wrapper" style={{ height: 240 }}>
          <Bar data={groupAvgData} options={groupAvgOpts} />
        </div>
      </div>

      {/* Summary table */}
      <div className="card">
        <div className="card-title">Side-by-side metric summary — all models, both datasets</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              <th>Dep Acc</th>
              <th>Dep F1</th>
              <th>Sui Acc</th>
              <th>Sui F1</th>
              <th>Dep AUC</th>
              <th>Sui AUC</th>
            </tr>
          </thead>
          <tbody>
            {depModels.map((dm) => {
              const sm = suiModels.find((s) => s.id === dm.id);
              return (
                <tr key={dm.id}>
                  <td>{dm.name}</td>
                  <td><span className={`badge ${dm.type === "DL" ? "dl" : "ml"}`}>{dm.type}</span></td>
                  <td style={{ fontFamily: "var(--font-mono)", color: PURPLE }}>{dm.accuracy.toFixed(1)}%</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: PURPLE }}>{dm.f1.toFixed(1)}%</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: CORAL }}>{sm ? sm.accuracy.toFixed(1) + "%" : "—"}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: CORAL }}>{sm ? sm.f1.toFixed(1) + "%" : "—"}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{dm.aucRoc.toFixed(3)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{sm ? sm.aucRoc.toFixed(3) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
