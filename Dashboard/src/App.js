import React, { useState } from "react";
import Overview from "./components/Overview";
import DatasetInfo from "./components/DatasetInfo";
import ModelResults from "./components/ModelResults";
import Comparison from "./components/Comparison";
import DeepLearning from "./components/DeepLearning";
import "./index.css";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "dataset", label: "Dataset Info" },
  { id: "models", label: "Model Results" },
  { id: "comparison", label: "Comparison" },
  { id: "deeplearning", label: "Deep Learning" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [activeDataset, setActiveDataset] = useState("depression");

  const renderTab = () => {
    switch (activeTab) {
      case "overview":     return <Overview activeDataset={activeDataset} />;
      case "dataset":      return <DatasetInfo activeDataset={activeDataset} />;
      case "models":       return <ModelResults activeDataset={activeDataset} />;
      case "comparison":   return <Comparison />;
      case "deeplearning": return <DeepLearning activeDataset={activeDataset} />;
      default:             return <Overview activeDataset={activeDataset} />;
    }
  };

  return (
    <div className="app">
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-left">
          <h1 className="topbar-title">Mental Health Detection — FYP Dashboard</h1>
          <p className="topbar-sub">
            Comparative Study: ML &amp; DL Approaches &nbsp;·&nbsp; NUCES Karachi
          </p>
        </div>
        <div className="dataset-toggle">
          <button
            className={`ds-btn dep ${activeDataset === "depression" ? "active" : ""}`}
            onClick={() => setActiveDataset("depression")}
          >
            Depression Dataset
          </button>
          <button
            className={`ds-btn sui ${activeDataset === "suicide" ? "active" : ""}`}
            onClick={() => setActiveDataset("suicide")}
          >
            Suicide Dataset
          </button>
        </div>
      </header>

      {/* Tab bar */}
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? "active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="content">{renderTab()}</main>
    </div>
  );
}
