"use client";

import React, { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";
import type { Message } from "@/types/chat";

interface Props {
  messages: Message[];
  isStreaming: boolean;
  onRate: (helpful: boolean) => void;
  onRegenerate: () => void;
  className?: string;
}

export function ChatWindow({ messages, isStreaming, onRate, onRegenerate, className }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  return (
    <div className={`flex-1 overflow-y-auto ${className || ""}`}>
      <div className="max-w-3xl mx-auto px-4 py-6">
        {messages.length === 0 ? (
          /* Welcome Screen */
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              DeepService 企业智能客服
            </h2>
            <p className="text-gray-500 dark:text-gray-400 mb-8 max-w-md">
              基于 DeepSeek 大模型 + RAG 检索增强生成<br />
              为企业提供精准、可靠的智能客服体验
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
              {[
                { icon: "📦", text: "我的订单发货了吗？" },
                { icon: "🔄", text: "如何申请退换货？" },
                { icon: "💰", text: "退款多久能到账？" },
                { icon: "👤", text: "VIP会员有哪些权益？" },
              ].map((item, i) => (
                <button
                  key={i}
                  className="flex items-center gap-2 px-4 py-3 text-sm text-left bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl hover:border-primary-300 hover:shadow-sm transition-all"
                  onClick={() => {
                    // 通过自定义事件发送推荐问题
                    window.dispatchEvent(
                      new CustomEvent("quick-question", { detail: item.text })
                    );
                  }}
                >
                  <span>{item.icon}</span>
                  <span className="text-gray-700 dark:text-gray-300">{item.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Messages */
          <>
            {messages.map((msg, i) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isLast={i === messages.length - 1 && !isStreaming}
                onRate={i === messages.length - 1 ? onRate : undefined}
                onRegenerate={i === messages.length - 1 ? onRegenerate : undefined}
              />
            ))}
          </>
        )}

        {/* Typing indicator for streaming */}
        {isStreaming && <TypingIndicator />}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
