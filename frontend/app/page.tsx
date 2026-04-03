"use client";

import { useState, useEffect, useRef } from "react";
import {
  Upload,
  BarChart3,
  History,
  Leaf,
  AlertCircle,
  CheckCircle,
} from "lucide-react";
import AnalyticsDashboard from "./components/AnalyticsDashboard";
import ImageUpload from "./components/ImageUpload";
import PredictionHistory from "./components/PredictionHistory";
import apiClient from "./services/api";

type TabType = "predict" | "analytics" | "history";

interface Prediction {
  disease: string;
  confidence: number;
  confidence_pct: string;
  severity: string;
  all_predictions: Record<string, number>;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("predict");
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "ok" | "error">(
    "checking",
  );
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    const checkApi = async () => {
      try {
        await apiClient.checkHealth();
        setApiStatus("ok");
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } catch (error) {
        console.error("API health check failed:", error);
        setApiStatus("error");
      }
    };
    const onFocus = () => {
      checkApi();
    };
    window.addEventListener("focus", onFocus);
    checkApi();
    intervalRef.current = window.setInterval(checkApi, 5000);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  const handlePredictionComplete = (prediction: Prediction) => {
    setLastPrediction(prediction);
    setApiStatus("ok");
    // Auto-switch to history tab to show result
    setActiveTab("history");
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case "critical":
        return "bg-red-500/20 border-red-500/50 text-red-100";
      case "high":
        return "bg-orange-500/20 border-orange-500/50 text-orange-100";
      case "moderate":
        return "bg-yellow-500/20 border-yellow-500/50 text-yellow-100";
      case "none":
        return "bg-green-500/20 border-green-500/50 text-green-100";
      default:
        return "bg-slate-500/20 border-slate-500/50 text-slate-100";
    }
  };

  const getSeverityIcon = (severity: string) => {
    if (severity.toLowerCase() === "none") {
      return <CheckCircle className="w-5 h-5" />;
    }
    return <AlertCircle className="w-5 h-5" />;
  };
  const diseaseBadge = (disease: string) => {
    const key = disease.toLowerCase();
    if (key.includes("mosaic"))
      return "bg-purple-500/20 text-purple-200 border border-purple-500/40";
    if (key.includes("bacterial"))
      return "bg-red-500/20 text-red-200 border border-red-500/40";
    if (key.includes("brown") || key.includes("streak"))
      return "bg-amber-500/20 text-amber-200 border border-amber-500/40";
    if (key.includes("green") || key.includes("mottle"))
      return "bg-emerald-500/20 text-emerald-200 border border-emerald-500/40";
    if (key.includes("healthy"))
      return "bg-green-500/20 text-green-200 border border-green-500/40";
    return "bg-slate-600/20 text-slate-200 border border-slate-500/40";
  };
  const confidenceBarColor = (v: number) => {
    if (v >= 0.7) return "bg-green-500";
    if (v >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };
  const confidenceTextColor = (v: number) => {
    if (v >= 0.7) return "text-green-400";
    if (v >= 0.5) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-800/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-green-400 to-emerald-600 rounded-lg">
              <Leaf className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Cassava Guard</h1>
              <p className="text-xs text-slate-400">Disease Detection AI</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {apiStatus === "ok" ? (
                <>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-green-400 hidden sm:inline">
                    API Connected
                  </span>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-red-400 hidden sm:inline">
                    {apiStatus === "checking" ? "Checking..." : "Offline"}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="border-b border-slate-700/50 bg-slate-800/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex gap-8">
          <button
            onClick={() => setActiveTab("predict")}
            className={`py-4 px-2 font-medium text-sm transition-colors border-b-2 ${
              activeTab === "predict"
                ? "border-green-400 text-green-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Upload className="w-4 h-4 inline mr-2" />
            Classify
          </button>
          <button
            onClick={() => setActiveTab("analytics")}
            className={`py-4 px-2 font-medium text-sm transition-colors border-b-2 ${
              activeTab === "analytics"
                ? "border-blue-400 text-blue-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Analytics
          </button>
          <button
            onClick={() => setActiveTab("history")}
            className={`py-4 px-2 font-medium text-sm transition-colors border-b-2 ${
              activeTab === "history"
                ? "border-purple-400 text-purple-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <History className="w-4 h-4 inline mr-2" />
            Results
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Predict Tab */}
        {activeTab === "predict" && (
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">
                Disease Diagnosis
              </h2>
              <p className="text-slate-300">
                Upload a cassava leaf image for instant AI diagnosis
              </p>
            </div>
            <ImageUpload onPredictionComplete={handlePredictionComplete} />

            {/* Latest Result Card */}
            {lastPrediction && (
              <div className="max-w-2xl mx-auto mt-8">
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
                  <h3 className="text-xl font-bold text-white mb-4">
                    Latest Diagnosis
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300 font-medium">
                        Condition:
                      </span>
                      <span
                        className={`text-sm font-semibold px-2 py-1 rounded ${diseaseBadge(lastPrediction.disease)}`}
                      >
                        {lastPrediction.disease}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300 font-medium">
                        Confidence:
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${confidenceBarColor(lastPrediction.confidence)}`}
                            style={{
                              width: `${lastPrediction.confidence * 100}%`,
                            }}
                          ></div>
                        </div>
                        <span
                          className={`font-bold ${confidenceTextColor(lastPrediction.confidence)}`}
                        >
                          {lastPrediction.confidence_pct}
                        </span>
                      </div>
                    </div>
                    <div
                      className={`flex items-center gap-2 p-3 rounded-lg border ${getSeverityColor(lastPrediction.severity)}`}
                    >
                      {getSeverityIcon(lastPrediction.severity)}
                      <span className="font-semibold capitalize">
                        Severity: {lastPrediction.severity}
                      </span>
                    </div>
                    <div className="space-y-3">
                      <p className="text-slate-300 font-medium">
                        Top Probabilities
                      </p>
                      {Object.entries(lastPrediction.all_predictions || {})
                        .sort((a, b) => (b[1] as number) - (a[1] as number))
                        .slice(0, 3)
                        .map(([name, prob]) => (
                          <div key={name} className="flex items-center gap-3">
                            <span
                              className={`text-xs font-semibold px-2 py-1 rounded ${diseaseBadge(name)}`}
                            >
                              {name}
                            </span>
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${confidenceBarColor(prob as number)}`}
                                style={{ width: `${(prob as number) * 100}%` }}
                              />
                            </div>
                            <span
                              className={`text-xs font-bold ${confidenceTextColor(prob as number)}`}
                            >
                              {((prob as number) * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === "analytics" && <AnalyticsDashboard />}

        {/* History Tab */}
        {activeTab === "history" && (
          <PredictionHistory lastPrediction={lastPrediction} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700/50 bg-slate-800/50 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-slate-400 text-sm">
          <p>
            Cassava Guard © 2026 | AI-Powered Agricultural Disease Detection
          </p>
          <p className="mt-2">
            Built with FastAI & Next.js for better crop health
          </p>
        </div>
      </footer>
    </div>
  );
}
