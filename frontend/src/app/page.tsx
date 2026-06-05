"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Menu } from "lucide-react";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { SessionPanel } from "@/components/chat/SessionPanel";
import { useChat } from "@/hooks/useChat";
import { useChatStore } from "@/stores/chat-store";
import apiClient from "@/lib/api";
import type { Conversation } from "@/types/chat";

export default function HomePage() {
  const store = useChatStore();
  const {
    messages,
    isStreaming,
    inputValue,
    setInputValue,
    sendMessage,
    cancelGeneration,
    regenerate,
    rateResponse,
  } = useChat();

  const [sidebarOpen, setSidebarOpen] = useState(false);

  // 加载会话列表
  const loadConversations = useCallback(async () => {
    try {
      const list = await apiClient.getConversations();
      store.setConversations(list as unknown as Conversation[]);
    } catch {
      // 后端未启动时静默
    }
  }, [store]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // 选择会话
  const handleSelectConversation = async (id: string) => {
    store.setCurrentId(id);
    store.reset();
    setSidebarOpen(false);
    try {
      const conv = await apiClient.getConversation(id);
      const msgs = (conv as unknown as { messages: unknown[] }).messages || [];
      store.setMessages(msgs as never[]);
    } catch {
      // fallback
    }
  };

  // 新建会话
  const handleNewConversation = () => {
    store.setCurrentId(null);
    store.reset();
    setSidebarOpen(false);
  };

  // 删除会话
  const handleDeleteConversation = async (id: string) => {
    try {
      await apiClient.deleteConversation(id);
      store.removeConversation(id);
    } catch {
      // silently fail
    }
  };

  // 推荐问题
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as string;
      if (detail) {
        setInputValue(detail);
        setTimeout(() => sendMessage(detail), 100);
      }
    };
    window.addEventListener("quick-question", handler);
    return () => window.removeEventListener("quick-question", handler);
  }, [sendMessage, setInputValue]);

  return (
    <div className="h-screen flex bg-white dark:bg-gray-950">
      {/* Session Sidebar */}
      <SessionPanel
        conversations={(store.conversations as unknown as Conversation[]) || []}
        currentId={store.currentId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onDelete={handleDeleteConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-14 border-b border-gray-200 dark:border-gray-800 flex items-center px-4 gap-3 flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <Menu size={20} />
          </button>
          <div className="flex-1">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-white truncate">
              {store.currentId
                ? (store.conversations as unknown as Conversation[]).find((c) => c.id === store.currentId)?.title ||
                  "对话"
                : "新对话"}
            </h2>
          </div>
          <div className="flex items-center gap-2">
            {store.isStreaming && (
              <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full animate-typing">
                生成中...
              </span>
            )}
          </div>
        </header>

        {/* Messages */}
        <ChatWindow
          messages={store.messages}
          isStreaming={store.isStreaming}
          streamContent={store.streamContent}
          onRate={(helpful) => rateResponse(helpful ? 5 : 1)}
          onRegenerate={regenerate}
        />

        {/* Input */}
        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={() => sendMessage()}
          isStreaming={isStreaming}
          onCancel={cancelGeneration}
        />
      </div>
    </div>
  );
}
