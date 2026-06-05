"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, ThumbsUp, ThumbsDown, Copy, RefreshCw } from "lucide-react";
import type { Message } from "@/types/chat";

interface Props {
  message: Message;
  isLast: boolean;
  onRate?: (helpful: boolean) => void;
  onRegenerate?: () => void;
}

export function MessageBubble({ message, isLast, onRate, onRegenerate }: Props) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const confidence = message.metadata?.confidence;

  if (isSystem) {
    return (
      <div className="flex justify-center my-2">
        <span className="text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 px-3 py-1 rounded-full">
          {message.content}
        </span>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 mb-4 animate-fade-in ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? "bg-primary-500 text-white"
            : "bg-gradient-to-br from-purple-500 to-blue-500 text-white"
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Bubble */}
      <div className={`max-w-[80%] min-w-[120px] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "bg-primary-500 text-white rounded-tr-sm"
              : "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-tl-sm shadow-sm"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}

          {/* Confidence badge */}
          {!isUser && confidence !== undefined && confidence < 0.7 && (
            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${
                  confidence < 0.4
                    ? "bg-red-100 text-red-600"
                    : "bg-yellow-100 text-yellow-600"
                }`}
              >
                {confidence < 0.4 ? "⚠️ 低置信度" : "⚡ 部分确认"}
              </span>
            </div>
          )}
        </div>

        {/* Actions */}
        {!isUser && message.content && isLast && (
          <div className="flex gap-1 mt-1 px-1">
            <button
              onClick={() => navigator.clipboard.writeText(message.content)}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-gray-600 transition-colors"
              title="复制"
            >
              <Copy size={14} />
            </button>
            {onRate && (
              <>
                <button
                  onClick={() => onRate(true)}
                  className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-green-500 transition-colors"
                  title="有帮助"
                >
                  <ThumbsUp size={14} />
                </button>
                <button
                  onClick={() => onRate(false)}
                  className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-red-500 transition-colors"
                  title="无帮助"
                >
                  <ThumbsDown size={14} />
                </button>
              </>
            )}
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-blue-500 transition-colors"
                title="重新生成"
              >
                <RefreshCw size={14} />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
