"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart3, MessageSquare, Database, Settings,
  TrendingUp, Users, Clock, AlertTriangle,
  ArrowUpRight, ArrowDownRight,
} from "lucide-react";
import apiClient from "@/lib/api";

interface DashboardStats {
  knowledge_base?: { total_chunks?: number };
  active_conversations?: number;
  model?: string;
  total_conversations?: number;
  avg_confidence?: number;
  transfer_rate?: number;
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<DashboardStats>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiClient
      .getStats()
      .then((s) => setStats(s as DashboardStats))
      .catch(() => setStats({}))
      .finally(() => setLoading(false));
  }, []);

  const cards = [
    {
      title: "活跃会话",
      value: stats.active_conversations || 0,
      change: "+12%",
      up: true,
      icon: MessageSquare,
      color: "bg-blue-50 text-blue-600",
    },
    {
      title: "知识库条目",
      value: stats.knowledge_base?.total_chunks || 0,
      change: "条 chunks",
      up: true,
      icon: Database,
      color: "bg-green-50 text-green-600",
    },
    {
      title: "平均置信度",
      value: stats.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "N/A",
      change: "",
      up: true,
      icon: TrendingUp,
      color: "bg-purple-50 text-purple-600",
    },
    {
      title: "转人工率",
      value: stats.transfer_rate ? `${(stats.transfer_rate * 100).toFixed(1)}%` : "N/A",
      change: "",
      up: false,
      icon: AlertTriangle,
      color: "bg-orange-50 text-orange-600",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-gray-400 hover:text-gray-600">← 返回对话</Link>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">管理后台</h1>
          </div>
          <span className="text-xs text-gray-500">
            模型: {stats.model || "DeepSeek"}
          </span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {cards.map((card, i) => (
            <div
              key={i}
              className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between mb-3">
                <div className={`w-10 h-10 rounded-xl ${card.color} flex items-center justify-center`}>
                  <card.icon size={20} />
                </div>
                {card.change && !card.change.includes("条") && (
                  <span
                    className={`flex items-center text-xs font-medium ${
                      card.up ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {card.up ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                    {card.change}
                  </span>
                )}
              </div>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {loading ? "—" : card.value}
              </p>
              <p className="text-sm text-gray-500 mt-1">{card.title}</p>
            </div>
          ))}
        </div>

        {/* Quick Links */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { title: "对话日志", desc: "查看和搜索历史对话记录", href: "/admin/conversations", icon: MessageSquare },
            { title: "知识库管理", desc: "管理文档、重新索引", href: "/admin/knowledge", icon: Database },
            { title: "数据分析", desc: "意图分布、满意度趋势", href: "/admin/analytics", icon: BarChart3 },
          ].map((link, i) => (
            <Link
              key={i}
              href={link.href}
              className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 hover:border-primary-300 hover:shadow-md transition-all group"
            >
              <div className="flex items-center gap-3 mb-3">
                <link.icon size={20} className="text-primary-500 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-gray-900 dark:text-white">{link.title}</h3>
              </div>
              <p className="text-sm text-gray-500">{link.desc}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
