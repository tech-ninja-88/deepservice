"use client";

import React from "react";
import { Plus, MessageSquare, Trash2, Search, X } from "lucide-react";
import type { Conversation } from "@/types/chat";
import { format } from "date-fns";
import { zhCN } from "date-fns/locale";

interface Props {
  conversations: Conversation[];
  currentId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export function SessionPanel({
  conversations,
  currentId,
  onSelect,
  onNew,
  onDelete,
  isOpen,
  onToggle,
}: Props) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={onToggle}
        />
      )}

      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-72 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-800">
          <div className="flex items-center justify-between mb-3">
            <h1 className="text-lg font-bold text-gray-900 dark:text-white">
              🤖 DeepService
            </h1>
            <button
              onClick={onToggle}
              className="lg:hidden p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <X size={18} />
            </button>
          </div>
          <button
            onClick={onNew}
            className="w-full flex items-center justify-center gap-2 bg-primary-500 hover:bg-primary-600 text-white rounded-xl py-2.5 text-sm font-medium transition-colors"
          >
            <Plus size={16} />
            新对话
          </button>
        </div>

        {/* Search */}
        <div className="px-4 py-2">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              placeholder="搜索会话..."
              className="w-full pl-8 pr-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm outline-none focus:border-primary-400"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-1">
          {conversations.length === 0 ? (
            <div className="text-center text-gray-400 text-sm mt-10">
              <MessageSquare size={32} className="mx-auto mb-2 opacity-50" />
              <p>暂无对话记录</p>
              <p className="text-xs mt-1">点击"新对话"开始</p>
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                onClick={() => onSelect(conv.id)}
                className={`group flex items-start gap-3 px-3 py-2.5 rounded-xl cursor-pointer mb-1 transition-colors ${
                  currentId === conv.id
                    ? "bg-primary-50 dark:bg-primary-900/30 border border-primary-200 dark:border-primary-800"
                    : "hover:bg-gray-100 dark:hover:bg-gray-800 border border-transparent"
                }`}
              >
                <MessageSquare
                  size={16}
                  className={`mt-0.5 flex-shrink-0 ${
                    currentId === conv.id ? "text-primary-500" : "text-gray-400"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {conv.title || "新对话"}
                  </p>
                  <p className="text-xs text-gray-500 truncate mt-0.5">
                    {conv.last_message || conv.message_count + " 条消息"}
                  </p>
                  <p className="text-[10px] text-gray-400 mt-0.5">
                    {format(new Date(conv.updated_at), "MM/dd HH:mm", { locale: zhCN })}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(conv.id);
                  }}
                  className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-100 dark:hover:bg-red-900/50 text-gray-400 hover:text-red-500 transition-all"
                  title="删除"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-200 dark:border-gray-800 text-center">
          <a
            href="/admin"
            className="text-xs text-gray-400 hover:text-primary-500 transition-colors"
          >
            管理后台
          </a>
        </div>
      </aside>
    </>
  );
}
