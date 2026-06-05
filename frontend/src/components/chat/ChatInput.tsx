"use client";

import React, { useRef, useEffect, KeyboardEvent } from "react";
import { Send, Square, Paperclip } from "lucide-react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  isStreaming: boolean;
  onCancel: () => void;
  placeholder?: string;
}

export function ChatInput({ value, onChange, onSend, isStreaming, onCancel, placeholder }: Props) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 150) + "px";
    }
  }, [value]);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isStreaming) onSend();
    }
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="max-w-3xl mx-auto flex items-end gap-2 bg-gray-50 dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 px-4 py-2 focus-within:border-primary-400 focus-within:ring-1 focus-within:ring-primary-400 transition-all">
        <button className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors hidden sm:block">
          <Paperclip size={18} />
        </button>

        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "输入您的问题... (Enter 发送, Shift+Enter 换行)"}
          rows={1}
          disabled={isStreaming}
          className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 py-1.5 max-h-[150px]"
        />

        {isStreaming ? (
          <button
            onClick={onCancel}
            className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-full transition-colors"
            title="停止生成"
          >
            <Square size={14} fill="white" />
          </button>
        ) : (
          <button
            onClick={onSend}
            disabled={!value.trim()}
            className="p-2 bg-primary-500 hover:bg-primary-600 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-full transition-colors"
            title="发送"
          >
            <Send size={14} />
          </button>
        )}
      </div>

      <p className="text-[10px] text-gray-400 text-center mt-2">
        DeepService 企业智能客服 · 基于 DeepSeek + RAG · 回复仅供参考
      </p>
    </div>
  );
}
