import axios, { AxiosInstance } from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

interface AnalyticsData {
  total_predictions: number;
  disease_breakdown: Array<{
    disease: string;
    count: number;
    avg_confidence: number;
  }>;
  recent_predictions: Array<{
    disease: string;
    confidence: number;
    time: string;
  }>;
}

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  async checkHealth() {
    try {
      const response = await this.client.get("/health");
      return response.data;
    } catch (error) {
      console.error("Health check failed:", error);
      throw error;
    }
  }

  async getInfo() {
    try {
      const response = await this.client.get("/info");
      return response.data;
    } catch (error) {
      console.error("Failed to get info:", error);
      throw error;
    }
  }

  async predict(
    file: File,
  ): Promise<{ success: boolean; prediction: Prediction; timestamp: string }> {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await this.client.post("/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data?.detail || "Prediction failed");
      }
      throw error;
    }
  }

  async getAnalytics(): Promise<AnalyticsData> {
    try {
      const response = await this.client.get("/analytics");
      return response.data;
    } catch (error) {
      console.error("Failed to fetch analytics:", error);
      throw error;
    }
  }

  async exportPredictions() {
    try {
      const response = await this.client.get("/predictions/export");
      return response.data;
    } catch (error) {
      console.error("Failed to export predictions:", error);
      throw error;
    }
  }

  async exportPredictionsXLSX(): Promise<Blob> {
    try {
      const response = await this.client.get("/predictions/export.xlsx", {
        responseType: "blob",
      });
      return response.data;
    } catch (error) {
      console.error("Failed to export XLSX:", error);
      throw error;
    }
  }

  async exportPredictionsPDF(): Promise<Blob> {
    try {
      const response = await this.client.get("/predictions/export.pdf", {
        responseType: "blob",
      });
      return response.data;
    } catch (error) {
      console.error("Failed to export PDF:", error);
      throw error;
    }
  }

  async clearPredictions() {
    try {
      const response = await this.client.delete("/predictions/clear");
      return response.data;
    } catch (error) {
      console.error("Failed to clear predictions:", error);
      throw error;
    }
  }
}

const apiClient = new APIClient();
export default apiClient;
