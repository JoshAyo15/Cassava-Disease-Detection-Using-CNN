"use client";

import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle, Download, Trash2, X } from "lucide-react";
import apiClient from "../services/api";

interface Prediction {
  id: number;
  disease: string;
  confidence: number;
  timestamp: string;
  image: string;
}

interface PredictionHistoryProps {
  lastPrediction?: unknown;
}

export default function PredictionHistory({
  lastPrediction,
}: PredictionHistoryProps) {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [showClearModal, setShowClearModal] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);
        const data = await apiClient.exportPredictions();
        setPredictions(data.data);
        setTotal(data.total);
      } catch (error) {
        console.error("Failed to load predictions:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [lastPrediction]);

  const handleExport = async () => {
    try {
      const data = await apiClient.exportPredictions();
      const json = JSON.stringify(data, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cassava-predictions-${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
    }
  };
  const handleExportXLSX = async () => {
    try {
      const blob = await apiClient.exportPredictionsXLSX();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cassava-predictions-${new Date().toISOString().split("T")[0]}.xlsx`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("XLSX export failed:", error);
    }
  };
  const handleExportPDF = async () => {
    try {
      const blob = await apiClient.exportPredictionsPDF();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cassava-predictions-${new Date().toISOString().split("T")[0]}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("PDF export failed:", error);
    }
  };

  const handleClear = async () => {
    setShowClearModal(true);
  };

  const formatDate = (datetime: string) => {
    return new Date(datetime).toLocaleString();
  };

  const getSeverityColor = (confidence: number) => {
    if (confidence >= 0.7) return "bg-green-500/20 border-green-500/50";
    if (confidence >= 0.5) return "bg-yellow-500/20 border-yellow-500/50";
    return "bg-red-500/20 border-red-500/50";
  };

  const getSeverityBadge = (confidence: number) => {
    if (confidence >= 0.7) return "High";
    if (confidence >= 0.5) return "Medium";
    return "Low";
  };
  const diseaseColor = (disease: string) => {
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

  return (
    <div className="space-y-6">
      {/* Header with Actions */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Prediction Results</h2>
          <p className="text-slate-400 text-sm mt-1">
            {total} {total === 1 ? "result" : "results"} found
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleExport}
            disabled={total === 0}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
          >
            <Download className="w-4 h-4" />
            JSON
          </button>
          <button
            onClick={handleExportXLSX}
            disabled={total === 0}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
          >
            <Download className="w-4 h-4" />
            Excel
          </button>
          <button
            onClick={handleExportPDF}
            disabled={total === 0}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
          >
            <Download className="w-4 h-4" />
            PDF
          </button>
          <button
            onClick={handleClear}
            disabled={total === 0}
            className="flex items-center gap-2 px-4 py-2 bg-red-600/50 hover:bg-red-600 text-red-100 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm border border-red-500/50"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>

      {/* Results */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin">
            <div className="w-8 h-8 border-4 border-slate-700 border-t-blue-400 rounded-full"></div>
          </div>
          <p className="text-slate-400 mt-4">Loading results...</p>
        </div>
      ) : total === 0 ? (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-12 text-center">
          <div className="inline-block p-4 bg-slate-700/50 rounded-full mb-4">
            <AlertCircle className="w-8 h-8 text-slate-400" />
          </div>
          <p className="text-slate-300 font-medium">No predictions yet</p>
          <p className="text-slate-400 text-sm mt-1">
            Upload a cassava leaf image to get started
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {predictions.map((pred, idx) => (
            <div
              key={idx}
              className={`bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border transition-all hover:border-slate-600 ${getSeverityColor(
                pred.confidence,
              )} p-6`}
            >
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {/* Disease Info */}
                <div className="md:col-span-2">
                  <div className="flex items-start gap-3">
                    <div className="mt-1">
                      {pred.confidence >= 0.7 ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-yellow-400" />
                      )}
                    </div>
                    <div>
                      <div
                        className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-semibold ${diseaseColor(pred.disease)}`}
                      >
                        {pred.disease}
                      </div>
                      <p className="text-slate-400 text-sm mt-1">
                        File: {pred.image}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Confidence */}
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-2">
                    Confidence
                  </p>
                  <div className="space-y-2">
                    <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`${pred.confidence >= 0.7 ? "h-full bg-green-500" : pred.confidence >= 0.5 ? "h-full bg-yellow-500" : "h-full bg-red-500"}`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                    <p
                      className={`text-2xl font-bold ${pred.confidence >= 0.7 ? "text-green-400" : pred.confidence >= 0.5 ? "text-yellow-400" : "text-red-400"}`}
                    >
                      {(pred.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Meta */}
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-2">
                    Confidence Level
                  </p>
                  <p className="text-sm font-semibold text-slate-200">
                    {getSeverityBadge(pred.confidence)}
                  </p>
                  <p className="text-xs text-slate-400 mt-3 uppercase tracking-wider">
                    timestamp
                  </p>
                  <p className="text-xs text-slate-300 font-mono">
                    {formatDate(pred.timestamp)}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination Info */}
      {predictions.length > 0 && (
        <div className="text-center py-4 text-slate-400 text-sm border-t border-slate-700/50 mt-6 pt-6">
          <p>
            Showing {predictions.length} of {total} results
          </p>
        </div>
      )}
      {showClearModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/60 z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-white font-semibold text-lg">
                Confirm Clear
              </h4>
              <button
                onClick={() => {
                  setShowClearModal(false);
                  setConfirmText("");
                }}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-slate-300 text-sm">
              This will permanently delete all prediction history. Type{" "}
              <span className="font-mono bg-slate-800 px-1 rounded">CLEAR</span>{" "}
              to confirm.
            </p>
            <input
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="Type CLEAR to confirm"
              className="mt-4 w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 outline-none focus:ring-2 focus:ring-red-500/50"
            />
            <div className="mt-6 flex justify-end gap-2">
              <button
                onClick={() => {
                  setShowClearModal(false);
                  setConfirmText("");
                }}
                className="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  if (confirmText !== "CLEAR") return;
                  try {
                    setClearing(true);
                    await apiClient.clearPredictions();
                    setPredictions([]);
                    setTotal(0);
                    setShowClearModal(false);
                    setConfirmText("");
                  } catch (e) {
                    console.error(e);
                  } finally {
                    setClearing(false);
                  }
                }}
                disabled={confirmText !== "CLEAR" || clearing}
                className="px-4 py-2 rounded bg-red-600 hover:bg-red-700 text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {clearing ? "Clearing..." : "Clear All"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
