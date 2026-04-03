import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cassava Disease Classifier",
  description: "AI-powered cassava leaf disease detection and analysis system",
  keywords: ["cassava", "disease", "classification", "agriculture", "ML"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
