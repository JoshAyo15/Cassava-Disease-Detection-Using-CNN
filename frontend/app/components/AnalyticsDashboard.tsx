"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  RadarChart,
} from "recharts";
import { Loader2, AlertTriangle } from "lucide-react";
import apiClient from "../services/api";

const COLORS = ["#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#8b5cf6"];

interface DiseaseData {
  disease: string;
  count: number;
  avg_confidence: number;
}

interface RecentPrediction {
  disease: string;
  confidence: number;
  time: string;
}

export default function AnalyticsDashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [diseaseData, setDiseaseData] = useState<DiseaseData[]>([]);
  const [recentData, setRecentData] = useState<RecentPrediction[]>([]);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setLoading(true);
        const data = await apiClient.getAnalytics();
        setTotalPredictions(data.total_predictions);
        setDiseaseData(data.disease_breakdown);
        setRecentData(data.recent_predictions);
        setError(null);
      } catch (err) {
        setError("Failed to load analytics data");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-red-400" />
          <div>
            <p className="font-semibold text-red-200">{error}</p>
            <p className="text-red-100 text-sm">No predictions made yet</p>
          </div>
        </div>
      </div>
    );
  }

  if (totalPredictions === 0) {
    return (
      <div className="text-center py-12">
        <div className="inline-block p-4 bg-blue-500/20 rounded-full mb-4">
          <AlertTriangle className="w-8 h-8 text-blue-400" />
        </div>
        <p className="text-slate-300 text-lg font-medium">No data yet</p>
        <p className="text-slate-400 text-sm">
          Make predictions to see analytics
        </p>
      </div>
    );
  }

  // Prepare chart data
  const chartData = diseaseData.map((item) => ({
    name: item.disease.replace("Cassava___", "").replace(/_/g, " "),
    value: item.count,
    confidence: (item.avg_confidence * 100).toFixed(1),
  }));

  const confidenceData = diseaseData.map((item) => ({
    name: item.disease.replace("Cassava___", "").replace(/_/g, " "),
    confidence: parseFloat((item.avg_confidence * 100).toFixed(1)),
  }));

  const radarData = diseaseData.map((item) => ({
    disease: item.disease.replace("Cassava___", "").replace(/_/g, " "),
    predictions: item.count,
    confidence: parseFloat((item.avg_confidence * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-8">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <p className="text-slate-400 text-sm font-medium">
            Total Predictions
          </p>
          <p className="text-3xl font-bold text-white mt-2">
            {totalPredictions}
          </p>
          <p className="text-green-400 text-xs mt-2">All time</p>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <p className="text-slate-400 text-sm font-medium">Most Common</p>
          <p className="text-2xl font-bold text-white mt-2">
            {diseaseData[0]?.disease
              .replace("Cassava___", "")
              .replace(/_/g, " ") || "N/A"}
          </p>
          <p className="text-blue-400 text-xs mt-2">
            {diseaseData[0]?.count || 0} cases
          </p>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <p className="text-slate-400 text-sm font-medium">Avg Confidence</p>
          <p className="text-3xl font-bold text-white mt-2">
            {(
              (diseaseData.reduce((sum, item) => sum + item.avg_confidence, 0) /
                diseaseData.length) *
              100
            ).toFixed(1)}
            %
          </p>
          <p className="text-orange-400 text-xs mt-2">Model accuracy</p>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <p className="text-slate-400 text-sm font-medium">
            Recent Prediction
          </p>
          <p className="text-sm font-mono text-white mt-2 break-all">
            {recentData[0]?.disease
              .replace("Cassava___", "")
              .replace(/_/g, " ") || "None yet"}
          </p>
          <p className="text-purple-400 text-xs mt-2">Just now</p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Prediction Distribution */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Prediction Distribution
          </h3>
          <ResponsiveContainer width="100%" height={360}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid #475569",
                  borderRadius: "0.5rem",
                  color: "#ffffff",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence Levels */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Average Confidence by Disease
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid #475569",
                  borderRadius: "0.5rem",
                  color: "#475569",
                }}
              />
              <Bar dataKey="confidence" radius={[8, 8, 0, 0]}>
                {confidenceData.map((entry, index) => (
                  <Cell
                    key={`conf-cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Case Count by Disease */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Cases by Disease
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid #475569",
                  borderRadius: "0.5rem",
                  color: "#475569",
                }}
              />
              <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`case-cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Prediction Metrics Overview
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#475569" />
              <PolarAngleAxis dataKey="disease" stroke="#94a3b8" />
              <PolarRadiusAxis stroke="#94a3b8" />
              <Radar
                name="Predictions"
                dataKey="predictions"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.6}
              />
              <Radar
                name="Confidence %"
                dataKey="confidence"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.6}
              />
              <Legend />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: "0.5rem",
                  color: "#ffffff",
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Predictions Table */}
      {recentData.length > 0 && (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Recent Predictions
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                    Disease
                  </th>
                  <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                    Confidence
                  </th>
                  <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                    Time
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentData.map((pred, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30"
                  >
                    <td className="py-3 px-4 text-slate-200">
                      {pred.disease
                        .replace("Cassava___", "")
                        .replace(/_/g, " ")}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`${pred.confidence >= 0.7 ? "h-full bg-green-500" : pred.confidence >= 0.5 ? "h-full bg-yellow-500" : "h-full bg-red-500"}`}
                            style={{ width: `${pred.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span
                          className={`${pred.confidence >= 0.7 ? "text-green-400" : pred.confidence >= 0.5 ? "text-yellow-400" : "text-red-400"} font-semibold text-xs`}
                        >
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-slate-400 text-xs">
                      {new Date(pred.time).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
