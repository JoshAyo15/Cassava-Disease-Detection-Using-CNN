"use client";

import { useRef, useState, useEffect } from "react";
import { Upload, Loader2, Brain, Zap, CheckCircle, Cpu } from "lucide-react";
import apiClient from "../services/api";

interface Prediction {
  disease: string;
  confidence: number;
  confidence_pct: string;
  severity: string;
  description: string;
  action: string;
  all_predictions: Record<string, number>;
  disease_folder: string;
}

interface ImageUploadProps {
  onPredictionComplete: (prediction: Prediction) => void;
}

type LoadingStage =
  | "sending"
  | "processing"
  | "analyzing"
  | "generating"
  | null;

export default function ImageUpload({
  onPredictionComplete,
}: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  // Simulate loading stages
  useEffect(() => {
    if (!loading) return;

    const stages: LoadingStage[] = [
      "sending",
      "processing",
      "analyzing",
      "generating",
    ];
    let stageIndex = 0;

    setLoadingStage(stages[0]);

    const interval = setInterval(() => {
      stageIndex = (stageIndex + 1) % stages.length;
      setLoadingStage(stages[stageIndex]);
    }, 1500);

    return () => clearInterval(interval);
  }, [loading]);

  const loadingMessages = {
    sending: "📤 Sending your leaf to the AI...",
    processing: "⚙️ Spinning up the neural networks...",
    analyzing: "🧠 AI is thinking hard right now...",
    generating: "✨ Crafting the perfect diagnosis...",
  };

  const loadingHumor = {
    sending: "Please don't move the leaf around while uploading 🍃",
    processing: "Our AI is having a coffee break... just kidding!",
    analyzing: "Warning: AI getting smarter by the second",
    generating: "Almost there... just double-checking for typos",
  };

  const getLoadingIcon = () => {
    switch (loadingStage) {
      case "sending":
        return <Zap className="w-12 h-12 text-yellow-400 animate-pulse" />;
      case "processing":
        return <Cpu className="w-12 h-12 text-blue-400 animate-spin" />;
      case "analyzing":
        return <Brain className="w-12 h-12 text-purple-400 animate-bounce" />;
      case "generating":
        return (
          <CheckCircle className="w-12 h-12 text-green-400 animate-pulse" />
        );
      default:
        return <Loader2 className="w-12 h-12 text-green-400 animate-spin" />;
    }
  };

  const handleFileSelect = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB");
      return;
    }

    setError(null);
    setLoading(true);
    setFileName(file.name);

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        setPreview(e.target.result as string);
      }
    };
    reader.readAsDataURL(file);

    try {
      const result = await apiClient.predict(file);
      if (result.success) {
        onPredictionComplete(result.prediction);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to classify image. Please try again.",
      );
      setPreview(null);
    } finally {
      setLoading(false);
      setLoadingStage(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add("border-green-400", "bg-green-400/10");
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove("border-green-400", "bg-green-400/10");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove("border-green-400", "bg-green-400/10");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Main Content */}
      {!loading || !preview ? (
        // Drop Zone (when not loading or no preview)
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed border-slate-500 rounded-lg p-12 text-center cursor-pointer transition-all ${
            loading
              ? "opacity-50 cursor-not-allowed"
              : "hover:border-green-400 hover:bg-green-400/5"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => {
              if (e.target.files?.[0]) {
                handleFileSelect(e.target.files[0]);
              }
            }}
            className="hidden"
            disabled={loading}
          />

          {loading && preview ? (
            // Loading with Preview State
            <div className="space-y-4">
              {getLoadingIcon()}
              <p className="text-slate-300 font-medium text-lg">
                {loadingStage ? loadingMessages[loadingStage] : "Analyzing..."}
              </p>
              <p className="text-slate-400 text-sm">
                {loadingStage ? loadingHumor[loadingStage] : "Processing..."}
              </p>
              <div className="flex justify-center mt-6">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" />
                  <div
                    className="w-2 h-2 bg-green-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  />
                  <div
                    className="w-2 h-2 bg-green-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  />
                </div>
              </div>
            </div>
          ) : (
            // Initial State
            <div className="space-y-4">
              <Upload className="w-12 h-12 text-slate-400 mx-auto" />
              <div>
                <p className="text-slate-200 font-semibold text-lg">
                  Drop image here
                </p>
                <p className="text-slate-400 text-sm">or click to browse</p>
              </div>
              <p className="text-xs text-slate-500">
                Supported formats: JPG, PNG | Max size: 10 MB
              </p>
            </div>
          )}
        </div>
      ) : null}

      {/* Loading with Image Preview - Split View */}
      {loading && preview && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* Image Preview */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 overflow-hidden p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">
                Image Review
              </h3>
              <img
                src={preview}
                alt="Preview"
                className="w-full h-64 object-cover rounded-lg border border-slate-600/50"
              />
              <p className="text-xs text-slate-400 mt-3 break-all">
                {fileName}
              </p>
            </div>

            {/* Loading Stage */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 p-8 flex flex-col justify-center items-center">
              <div className="text-center space-y-6">
                {getLoadingIcon()}

                <div className="space-y-3">
                  <p className="text-slate-200 font-semibold text-lg">
                    {loadingStage
                      ? loadingMessages[loadingStage]
                      : "Analyzing..."}
                  </p>
                  <p className="text-slate-400 text-sm">
                    {loadingStage
                      ? loadingHumor[loadingStage]
                      : "Processing image..."}
                  </p>
                </div>

                {/* Animated dots */}
                <div className="flex justify-center gap-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-bounce" />
                  <div
                    className="w-3 h-3 bg-green-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  />
                  <div
                    className="w-3 h-3 bg-green-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  />
                </div>

                {/* Progress indicator */}
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-gradient-to-r from-green-400 to-emerald-600 h-2 rounded-full w-3/4 animate-pulse" />
                </div>

                <p className="text-xs text-slate-500 mt-4">
                  Typically takes 1-2 seconds on CPU, faster on GPU
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results - shown after loading completes */}
      {preview && !loading && !error && (
        <div className="mt-8 bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700/50 overflow-hidden p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Image Preview */}
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2">
                Classified Image
              </h3>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Preview"
                className="w-full h-64 object-cover rounded-lg border border-slate-600/50"
              />
              <p className="text-xs text-slate-400 mt-3 break-all">
                {fileName}
              </p>
            </div>

            {/* Info */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-300">
                Classification Complete
              </h3>
              <div className="bg-slate-700/50 rounded-lg p-4 space-y-3">
                <p className="text-sm text-slate-300">
                  ✓ Image successfully analyzed by the AI model
                </p>
                <p className="text-sm text-slate-300">
                  ✓ Check the &quot;Results&quot; tab to see detailed
                  predictions
                </p>
                <p className="text-sm text-slate-300">
                  ✓ View the &quot;Analytics&quot; tab to see aggregated
                  statistics
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="mt-8 bg-red-500/20 border border-red-500/50 rounded-lg p-6">
          <div className="space-y-4">
            <p className="font-semibold text-red-200">
              ❌ Classification Failed
            </p>
            <p className="text-red-100 text-sm">{error}</p>
            <button
              onClick={() => {
                setPreview(null);
                setError(null);
                setFileName(null);
                fileInputRef.current?.click();
              }}
              className="mt-4 px-4 py-2 bg-red-500/30 hover:bg-red-500/40 text-red-100 rounded-lg transition-colors text-sm"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {preview && !loading && (
        <div className="mt-6 flex gap-4 justify-center">
          <button
            onClick={() => {
              setPreview(null);
              setError(null);
              setFileName(null);
            }}
            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-slate-100 rounded-lg transition-colors font-medium text-sm"
          >
            Clear
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors font-medium text-sm"
          >
            Upload New
          </button>
        </div>
      )}
    </div>
  );
}
