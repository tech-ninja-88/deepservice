"use client";

import React from "react";
import Link from "next/link";
import { ArrowLeft, BarChart3, TrendingUp, PieChart, Activity } from "lucide-react";

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4">
        <div className="flex items-center gap-4 max-w-7xl mx-auto">
          <Link href="/admin" className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"><ArrowLeft size={18} /></Link>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">数据分析</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { title: "意图分布", desc: "各意图类型的占比趋势", icon: PieChart, placeholder: "连接后端 API 后显示意图分布图表" },
            { title: "满意度趋势", desc: "用户满意度评分变化", icon: TrendingUp, placeholder: "连接后端 API 后显示满意度趋势图" },
            { title: "响应时间", desc: "API 响应延迟分布", icon: Activity, placeholder: "连接后端 API 后显示响应时间分布" },
            { title: "转人工率", desc: "转人工比例变化趋势", icon: BarChart3, placeholder: "连接后端 API 后显示转人工率趋势" },
          ].map((item, i) => (
            <div key={i} className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-8 text-center">
              <item.icon size={40} className="mx-auto mb-3 text-gray-300" />
              <h3 className="font-semibold text-gray-900 dark:text-white mb-1">{item.title}</h3>
              <p className="text-sm text-gray-500 mb-2">{item.desc}</p>
              <p className="text-xs text-gray-400">{item.placeholder}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
