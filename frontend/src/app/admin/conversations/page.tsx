"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Search, MessageSquare, Eye } from "lucide-react";
import apiClient from "@/lib/api";

export default function ConversationsPage() {
  const [logs, setLogs] = useState<Record<string, unknown>[]>([]);

  useEffect(() => {
    apiClient.getLogs(1, 20).then((res) => {
      setLogs((res as { items?: Record<string, unknown>[] })?.items || []);
    }).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4">
        <div className="flex items-center gap-4 max-w-7xl mx-auto">
          <Link href="/admin" className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"><ArrowLeft size={18} /></Link>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">对话日志</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">会话 ID</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">用户</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">状态</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">消息数</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">创建时间</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600 dark:text-gray-400">操作</th>
                </tr>
              </thead>
              <tbody>
                {logs.length > 0 ? (
                  logs.map((log, i) => (
                    <tr key={i} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/30">
                      <td className="px-4 py-3 font-mono text-xs">{(log.id as string)?.slice(0, 12)}...</td>
                      <td className="px-4 py-3">{(log.user_id as string) || "匿名"}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs ${
                          log.status === "active" ? "bg-green-100 text-green-700" :
                          log.status === "closed" ? "bg-gray-100 text-gray-500" :
                          "bg-yellow-100 text-yellow-700"
                        }`}>
                          {log.status as string || "—"}
                        </span>
                      </td>
                      <td className="px-4 py-3">{log.message_count as number || 0}</td>
                      <td className="px-4 py-3 text-gray-500 text-xs">{log.created_at as string || "—"}</td>
                      <td className="px-4 py-3">
                        <button className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-400"><Eye size={16} /></button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="px-4 py-12 text-center text-gray-400">
                      <MessageSquare size={32} className="mx-auto mb-2 opacity-50" />
                      <p>暂无对话记录</p>
                      <p className="text-xs mt-1">连接后端 API 后显示对话日志</p>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
