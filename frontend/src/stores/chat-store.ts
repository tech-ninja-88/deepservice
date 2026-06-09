/**
 * Chat Store — 全局状态管理 (Zustand)
 */
import { create } from "zustand";
import type { Message, Conversation } from "@/types/chat";

interface StreamingState {
  isStreaming: boolean;
  streamContent: string;
}

interface ChatState {
  conversations: Conversation[];
  currentId: string | null;
  messages: Message[];
  isStreaming: boolean;
  streamContent: string;
  error: string | null;
  selectedRating: number | null;

  setConversations: (list: Conversation[]) => void;
  setCurrentId: (id: string | null) => void;
  setMessages: (msgs: Message[]) => void;
  addMessage: (msg: Message) => void;
  updateLastAssistant: (content: string) => void;
  setStreaming: (s: StreamingState) => void;
  setError: (e: string | null) => void;
  setRating: (r: number | null) => void;
  removeConversation: (id: string) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentId: null,
  messages: [],
  isStreaming: false,
  streamContent: "",
  error: null,
  selectedRating: null,

  setConversations: (list) => set({ conversations: list }),
  setCurrentId: (id) => set({ currentId: id }),
  setMessages: (msgs) => set({ messages: msgs }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  updateLastAssistant: (token) =>
    set((s) => {
      const msgs = [...s.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          // 防止相同 token 被重复追加
          if (msgs[i].content.endsWith(token)) continue;
          msgs[i] = { ...msgs[i], content: msgs[i].content + token };
          break;
        }
      }
      return { messages: msgs };
    }),
  setStreaming: (st) =>
    set({ isStreaming: st.isStreaming, streamContent: st.streamContent }),
  setError: (e) => set({ error: e }),
  setRating: (r) => set({ selectedRating: r }),
  removeConversation: (id) =>
    set((s) => ({
      conversations: s.conversations.filter((c) => c.id !== id),
      currentId: s.currentId === id ? null : s.currentId,
    })),
  reset: () =>
    set({ messages: [], isStreaming: false, streamContent: "", error: null }),
}));
