import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepService — 企业级智能客服",
  description: "基于 DeepSeek 大模型 + RAG 的企业级智能客服系统",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">{children}</body>
    </html>
  );
}
