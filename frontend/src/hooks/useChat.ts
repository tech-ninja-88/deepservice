/**
 * useChat — 核心对话 Hook
 * 管理流式消息收发、会话生命周期
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { useChatStore } from "@/stores/chat-store";
import apiClient, { ApiClient } from "@/lib/api";
import type { Message, SSETokenEvent } from "@/types/chat";

export function useChat() {
  const store = useChatStore();
  const abortRef = useRef<AbortController | null>(null);
  const [inputValue, setInputValue] = useState("");

  /** 发送消息（流式） */
  const sendMessage = useCallback(
    async (text?: string) => {
      const content = text || inputValue.trim();
      if (!content || store.isStreaming) return;

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
        const stream = await apiClient.chatStream(
          content,
          store.currentId || undefined,
          abortRef.current?.signal
        );

        const fullResponse: { metadata: Record<string, unknown> | null } = {
          metadata: null,
        };

        ApiClient.parseSSEStream(
          stream,
          // onToken
          (token) => {
            store.updateLastAssistant(token);
            store.setStreaming({
              isStreaming: true,
              streamContent: store.streamContent + token,
            });
          },
          // onEvent
          (event: SSETokenEvent) => {
            if (event.type === "metadata" && event.data) {
              fullResponse.metadata = event.data as Record<string, unknown>;
              const convId = (event.data as Record<string, unknown>).conversation_id as string;
              if (convId && !store.currentId) {
                store.setCurrentId(convId);
              }
            }
          },
          // onError
          (err) => {
            store.setError(err.message);
            store.updateLastAssistant(`\n\n⚠️ 抱歉，回复生成失败：${err.message}`);
          },
          // onDone
          () => {
            store.setStreaming({ isStreaming: false, streamContent: "" });
            if (store.currentId) {
              apiClient.getConversations().then(store.setConversations).catch(() => {});
            }
          }
        );

        abortRef.current = abortRef.current;
      } catch (err) {
        store.setError(err instanceof Error ? err.message : "Unknown error");
        store.setStreaming({ isStreaming: false, streamContent: "" });
      }
    },
    [inputValue, store]
  );

  /** 取消当前生成 */
  const cancelGeneration = useCallback(() => {
    abortRef.current?.abort();
    store.setStreaming({ isStreaming: false, streamContent: "" });
    abortRef.current = null;
  }, [store]);

  /** 重新生成 */
  const regenerate = useCallback(() => {
    const msgs = store.messages;
    if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
      const newMsgs = msgs.slice(0, -1);
      store.setMessages(newMsgs);
    }
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
