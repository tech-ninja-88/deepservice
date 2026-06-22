/** Core chat hook — manages streaming messages and session lifecycle. */
"use client";

import { useCallback, useRef, useState } from "react";
import { useChatStore } from "@/stores/chat-store";
import apiClient, { ApiClient } from "@/lib/api";
import type { Message, SSETokenEvent } from "@/types/chat";

export function useChat() {
  // Atomic selectors — each returns a stable reference unless its slice changes.
  const messages = useChatStore((s) => s.messages);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const streamContent = useChatStore((s) => s.streamContent);
  const error = useChatStore((s) => s.error);
  const currentId = useChatStore((s) => s.currentId);
  const selectedRating = useChatStore((s) => s.selectedRating);

  const abortRef = useRef<AbortController | null>(null);
  const [inputValue, setInputValue] = useState("");

  /** Send a user message and stream the assistant response. */
  const sendMessage = useCallback(
    async (text?: string) => {
      const content = text || inputValue.trim();
      if (!content) return;

      const {
        isStreaming: streaming,
        currentId: id,
        addMessage,
        setStreaming,
        setError,
        updateLastAssistant,
        setCurrentId,
        setConversations,
      } = useChatStore.getState();

      if (streaming) return;
      setInputValue("");

      // Add user message
      const userMsg: Message = {
        id: `user_${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMsg);

      // Placeholder assistant message
      const asstMsg: Message = {
        id: `asst_${Date.now()}`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      };
      addMessage(asstMsg);
      setStreaming({ isStreaming: true, streamContent: "" });

      try {
        const abortController = new AbortController();
        abortRef.current = abortController;

        const stream = await apiClient.chatStream(
          content,
          id || undefined,
          abortController.signal
        );

        ApiClient.parseSSEStream(
          stream,
          // onToken
          (token) => {
            const { updateLastAssistant: upd, setStreaming: setStr, streamContent: sc } =
              useChatStore.getState();
            upd(token);
            setStr({ isStreaming: true, streamContent: sc + token });
          },
          // onEvent
          (event: SSETokenEvent) => {
            if (event.type === "metadata" && event.data) {
              const convId = (event.data as Record<string, unknown>).conversation_id as string;
              if (convId) {
                const { currentId: cid, setCurrentId: setCid } = useChatStore.getState();
                if (!cid) setCid(convId);
              }
            }
          },
          // onError
          (err) => {
            useChatStore.getState().setError(err.message);
            useChatStore.getState().updateLastAssistant(
              `\n\n⚠️ Sorry, response generation failed: ${err.message}`
            );
          },
          // onDone
          () => {
            const { setStreaming: done, currentId: cid, setConversations: setConvs } =
              useChatStore.getState();
            done({ isStreaming: false, streamContent: "" });
            if (cid) {
              apiClient.getConversations().then(setConvs).catch(() => {});
            }
          }
        );
      } catch (err) {
        useChatStore.getState().setError(
          err instanceof Error ? err.message : "Unknown error"
        );
        useChatStore.getState().setStreaming({ isStreaming: false, streamContent: "" });
      }
    },
    [inputValue]
  );

  /** Cancel the current streaming generation. */
  const cancelGeneration = useCallback(() => {
    abortRef.current?.abort();
    useChatStore.getState().setStreaming({ isStreaming: false, streamContent: "" });
    abortRef.current = null;
  }, []);

  /** Regenerate the last assistant response. */
  const regenerate = useCallback(() => {
    const { messages: msgs, setMessages } = useChatStore.getState();
    // Find the last user message before mutating state
    const lastUserMsg = [...msgs].reverse().find((m) => m.role === "user");
    if (!lastUserMsg) return;

    // Remove last assistant bubble if present
    if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
      setMessages(msgs.slice(0, -1));
    }

    sendMessage(lastUserMsg.content);
  }, [sendMessage]);

  /** Submit a rating for the current conversation. */
  const rateResponse = useCallback(
    async (rating: number, comment?: string) => {
      const { currentId: id, setRating } = useChatStore.getState();
      if (!id) return;
      setRating(rating);
      try {
        await apiClient.rateConversation(id, rating, comment);
      } catch {
        // fire-and-forget; non-critical
      }
    },
    []
  );

  return {
    messages,
    isStreaming,
    streamContent,
    error,
    inputValue,
    setInputValue,
    sendMessage,
    cancelGeneration,
    regenerate,
    rateResponse,
    selectedRating,
    conversationId: currentId,
  };
}
