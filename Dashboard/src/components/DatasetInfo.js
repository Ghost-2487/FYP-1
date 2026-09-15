import React, { useState } from "react";
import { Pie, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, ArcElement, CategoryScale, LinearScale,
  BarElement, Tooltip, Legend,
} from "chart.js";
import { depressionDataset } from "../data/depressionData";
import { suicideDataset } from "../data/suicideData";

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export default function DatasetInfo({ activeDataset }) {
  const isSui = activeDataset === "suicide";
  const dataset = isSui ? suicideDataset : depressionDataset;
  const accent  = isSui ? "#e8714a" : "#7c71e8";
  const accentL = isSui ? "#f5a07a" : "#a89ef5";

  const pieData = {
    labels: Object.keys(dataset.classDistribution),
    datasets: [{
      data: Object.values(dataset.classDistribution),
      backgroundColor: [accent, "rgba(255,255,255,0.10)"],
      borderColor: ["transparent", "transparent"],
    }],
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw}%` } },
    },
  };

  const textLenData = {
    labels: ["0-20", "21-40", "41-60", "61-80", "81-100", "101-120", "121-140", "140+"],
    datasets: [{
      label: "Posts",
      // PLACEHOLDER distribution — replace with actual histogram counts from your data
      data: isSui ? [759, 4502, 9123, 12801, 13815, 12608, 11419, 177039] : [168579, 349413, 327119, 259450, 207594, 193885, 93200, 760],
      backgroundColor: accentL + "bb",
      borderRadius: 3,
      borderSkipped: false,
    }],
  };

  const textLenOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw.toLocaleString()} posts` } } },
    scales: {
      x: { ticks: { color: "#9999aa", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { ticks: { color: "#9999aa", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  };

  return (
    <div>
      <div style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 10 }}>
        <span className={`badge ${isSui ? "sui" : "dep"}`}>{dataset.name}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{dataset.platform}</span>
      </div>

      {/* Description */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
          {dataset.description}
        </div>
        {isSui && dataset.sources && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>Source breakdown</div>
            <div className="source-bar">
              {dataset.sources.map((s, i) => (
                <div
                  key={s.name}
                  style={{
                    width: s.percentage + "%",
                    background: i === 0 ? "#e8714a99" : "#7c71e899",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 12, color: "#fff", fontWeight: 500,
                  }}
                >
                  {s.name} {s.percentage.toFixed(0)}%
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 20, marginTop: 4 }}>
              {dataset.sources.map((s) => (
                <span key={s.name} style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  {s.name}: {s.samples.toLocaleString()} samples
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid3">
        {/* Key stats */}
        <div className="card">
          <div className="card-title">Dataset statistics</div>
          <div className="info-row"><span className="key">Total samples</span><span className="val">{dataset.totalSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Train</span><span className="val">{dataset.trainSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Validation</span><span className="val">{dataset.valSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Test</span><span className="val">{dataset.testSamples.toLocaleString()}</span></div>
          <div className="info-row"><span className="key">Split ratio</span><span className="val">{dataset.splitRatio}</span></div>
          <div className="info-row"><span className="key">Avg text length</span><span className="val">~{dataset.avgTextLength} tokens</span></div>
          <div className="info-row"><span className="key">Max text length</span><span className="val">{dataset.maxTextLength} tokens</span></div>
          <div className="info-row"><span className="key">Vocabulary size</span><span className="val">{dataset.vocabularySize.toLocaleString()}</span></div>
        </div>

        {/* Class distribution pie */}
        <div className="card">
          <div className="card-title">Class distribution</div>
          <div style={{ height: 160, position: "relative", marginBottom: 12 }}>
            <Pie data={pieData} options={pieOptions} />
          </div>
          {Object.entries(dataset.classCounts).map(([cls, count], i) => (
            <div key={cls} className="info-row">
              <span className="key" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: i === 0 ? accent : "rgba(255,255,255,0.15)", display: "inline-block" }} />
                {cls}
              </span>
              <span className="val">{count.toLocaleString()} ({dataset.classDistribution[cls]}%)</span>
            </div>
          ))}
        </div>

        {/* Preprocessing */}
        <div className="card">
          <div className="card-title">Preprocessing pipeline</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {dataset.preprocessingSteps.map((step, i) => (
              <div key={step} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{
                  width: 22, height: 22, borderRadius: "50%",
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid var(--border)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, color: "var(--text-muted)", flexShrink: 0,
                }}>{i + 1}</div>
                <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{step}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Text length distribution */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">Text length distribution (tokens)</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
          Replace with actual histogram data from your dataset analysis
        </div>
        <div className="chart-wrapper" style={{ height: 500 }}>
          <Bar data={textLenData} options={textLenOptions} />
        </div>
      </div>

      {/* Word clouds */}
      <div className="grid2">
        <div className="card">
          <div className="card-title">Word cloud — {dataset.classes[0]}</div>
          <WordCloudDisplay src={isSui ? dataset.wordCloudSuicidal : dataset.wordCloudDepressed} label={dataset.classes[0]} accent={accent} />
        </div>
        <div className="card">
          <div className="card-title">Word cloud — {dataset.classes[1]}</div>
          <WordCloudDisplay src={isSui ? dataset.wordCloudNotSuicidal : dataset.wordCloudNotDepressed} label={dataset.classes[1]} accent="rgba(255,255,255,0.3)" />
        </div>
      </div>
    </div>
  );
}

function WordCloudDisplay({ src, label, accent }) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError]   = useState(false);
  return (
    <>
      {!error
        ? <img
            src={src}
            alt={`Word cloud for ${label}`}
            className="wc-img"
            onLoad={() => setLoaded(true)}
            onError={() => setError(true)}
            style={{ display: loaded ? "block" : "none", width: "100%", height: "auto", objectFit: "contain" }}
          />
        : null}
      {(!loaded || error) && (
        <div className="wc-placeholder">
          <div style={{ fontSize: 28, color: accent, opacity: 0.4, userSelect: "none" }}>☁</div>
          <p>Place your word cloud image in</p>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>public/images/</p>
          <p>and update the path in the data file</p>
        </div>
      )}
    </>
  );
}
