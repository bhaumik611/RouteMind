import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RouteMind — Adaptive LLM Gateway & Routing Intelligence",
  description: "Enterprise LLM Gateway with Intelligent Cost-, Quality-, Latency-, and Reliability-Aware Model Routing",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-gray-100 min-h-screen antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
