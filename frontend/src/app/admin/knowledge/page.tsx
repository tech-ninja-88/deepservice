"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Search, Plus, FileText, Trash2, RefreshCw, BookOpen } from "lucide-react";
import apiClient from "@/lib/api";

interface KnowledgeDoc {
  id?: string;
  title?: string;
  category?: string;
  status?: string;
  updated_at?: string;
}

export default function KnowledgePage() {
  const [docs, setDocs] = useState<KnowledgeDoc[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<unknown[]>([]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const res = await apiClient.searchKnowledge(searchQuery, 10);
      setSearchResults((res as { results?: unknown[] })?.results || []);
    } catch {
      setSearchResults([]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4">
        <div className="flex items-center gap-4 max-w-7xl mx-auto">
          <Link href="/admin" className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800">
            <ArrowLeft size={18} />
          </Link>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">知识库管理</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        {/* Search */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Search size={18} /> 知识库检索
          </h2>
          <div className="flex gap-2">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="输入查询内容..."
              className="flex-1 px-4 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm outline-none focus:border-primary-400"
            />
            <button
              onClick={handleSearch}
              className="px-6 py-2.5 bg-primary-500 hover:bg-primary-600 text-white rounded-xl text-sm font-medium transition-colors"
            >
              搜索
            </button>
          </div>

          {searchResults.length > 0 && (
            <div className="mt-4 space-y-2">
              {searchResults.map((r: unknown, i: number) => {
                const result = r as Record<string, unknown>;
                return (
                  <div key={i} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-xl text-sm">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-primary-500 font-medium">
                        相关度: {((result.score as number) || 0).toFixed(2)}
                      </span>
                      <span className="text-xs text-gray-400">
                        {(result.metadata as Record<string, unknown>)?.title as string || ""}
                      </span>
                    </div>
                    <p className="text-gray-700 dark:text-gray-300 text-sm">
                      {(result.content as string)?.slice(0, 300) || ""}...
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {[
            { label: "文档总数", value: "—", icon: FileText },
            { label: "知识块", value: "—", icon: BookOpen },
            { label: "最后更新", value: "—", icon: RefreshCw },
          ].map((stat, i) => (
            <div key={i} className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
              <div className="flex items-center gap-3">
                <stat.icon size={20} className="text-primary-500" />
                <div>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className="text-xs text-gray-500">{stat.label}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Placeholder for document management */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-8 text-center">
          <BookOpen size={40} className="mx-auto mb-3 text-gray-300" />
          <p className="text-gray-500">连接后端 API 后可管理知识库文档</p>
          <p className="text-xs text-gray-400 mt-1">支持 Markdown / PDF / Word 文档导入</p>
        </div>
      </div>
    </div>
  );
}
