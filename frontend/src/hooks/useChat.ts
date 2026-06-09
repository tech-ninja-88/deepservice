/**
 * useChat — 核心对话 Hook
 * 管理流式消息收发、会话生命周期
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { useChatStore } from "@/stores/chat-store";
import apiClient from "@/lib/api";
import type { Message } from "@/types/chat";

export function useChat() {
  const store = useChatStore();
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false);
  const [inputValue, setInputValue] = useState("");

  /** 发送消息（非流式，避免 SSE 解析问题） */
  const sendMessage = useCallback(
    async (text?: string) => {
      if (sendingRef.current) return;
      const content = text || inputValue.trim();
      if (!content || store.isStreaming) return;
      sendingRef.current = true;

      setInputValue("");

      // 添加用户消息
      const userMsg: Message = {
        id: `user_${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      store.addMessage(userMsg);

      // 占位的 assistant 消息
      const asstMsg: Message = {
        id: `asst_${Date.now()}`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      };
      store.addMessage(asstMsg);
      store.setStreaming({ isStreaming: true, streamContent: "" });

      try {
        const response = await apiClient.chat(content, store.currentId || undefined);

        // 更新 conversation_id
        if (response.conversation_id && !store.currentId) {
          store.setCurrentId(response.conversation_id);
        }

        // 一次性替换 assistant 消息内容
        store.updateLastAssistant(response.content);

        sendingRef.current = false;
        store.setStreaming({ isStreaming: false, streamContent: "" });

        // 刷新会话列表
        if (store.currentId) {
          apiClient.getConversations().then(store.setConversations).catch(() => {});
        }
      } catch (err) {
        sendingRef.current = false;
        store.setError(err instanceof Error ? err.message : "Unknown error");
        store.setStreaming({ isStreaming: false, streamContent: "" });
      }
    },
    [inputValue, store]
  );

  /** 取消当前生成 */
  const cancelGeneration = useCallback(() => {
    abortRef.current?.abort();
    sendingRef.current = false;
    store.setStreaming({ isStreaming: false, streamContent: "" });
    abortRef.current = null;
  }, [store]);

  /** 重新生成 */
  const regenerate = useCallback(() => {
    const msgs = store.messages;
    // 移除最后一条 assistant 消息
    if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
      const newMsgs = msgs.slice(0, -1);
      store.setMessages(newMsgs);
    }
    // 重新发送上一条用户消息
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        sendMessage(msgs[i].content);
        break;
      }
    }
  }, [store, sendMessage]);

  /** 评价 */
  const rateResponse = useCallback(
    async (rating: number, comment?: string) => {
      if (!store.currentId) return;
      store.setRating(rating);
      try {
        await apiClient.rateConversation(store.currentId, rating, comment);
      } catch {
        // silently fail
      }
    },
    [store]
  );

  return {
    messages: store.messages,
    isStreaming: store.isStreaming,
    streamContent: store.streamContent,
    error: store.error,
    inputValue,
    setInputValue,
    sendMessage,
    cancelGeneration,
    regenerate,
    rateResponse,
    selectedRating: store.selectedRating,
    conversationId: store.currentId,
  };
}
