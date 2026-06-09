/**
 * DeepService API Client
 * 统一管理所有 API 请求，支持流式 (SSE) 和普通请求
 */

import type { ChatResponse, Conversation, SSETokenEvent } from "@/types/chat";

let _cachedBase: string | null = null;

async function getApiBase(): Promise<string> {
  if (_cachedBase) return _cachedBase;
  // 运行时从 /api/config 读取后端地址（无需重新构建镜像）
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    if (cfg.apiUrl) {
      _cachedBase = cfg.apiUrl;
      return _cachedBase;
    }
  } catch { /* fallback */ }
  _cachedBase = process.env.NEXT_PUBLIC_API_URL || "";
  return _cachedBase;
}

class ApiClient {
  private baseUrl: string | null = null;

  private async getBaseUrl(): Promise<string> {
    if (!this.baseUrl) {
      this.baseUrl = (await getApiBase()).replace(/\/$/, "");
    }
    return this.baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const base = await this.getBaseUrl();
    const url = `${base}${path}`;
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`API Error ${res.status}: ${err}`);
    }
    return res.json();
  }

  /** 流式对话 — 返回 ReadableStream 用于 SSE */
  async chatStream(
    message: string,
    conversationId?: string,
    signal?: AbortSignal
  ): Promise<ReadableStream<Uint8Array>> {
    const base = await this.getBaseUrl();
    const res = await fetch(`${base}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: conversationId, stream: true }),
      signal,
    });
    if (!res.ok) throw new Error(`Stream error ${res.status}`);
    return res.body!;
  }

  /** 非流式对话 */
  async chat(message: string, conversationId?: string): Promise<ChatResponse> {
    return this.request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, conversation_id: conversationId, stream: false }),
    });
  }

  /** 获取会话列表 */
  async getConversations(): Promise<Conversation[]> {
    return this.request<Conversation[]>("/api/conversations");
  }

  /** 获取会话详情 */
  async getConversation(id: string): Promise<Conversation & { messages: unknown[] }> {
    return this.request(`/api/conversations/${id}`);
  }

  /** 删除会话 */
  async deleteConversation(id: string): Promise<void> {
    await this.request(`/api/conversations/${id}`, { method: "DELETE" });
  }

  /** 提交评价 */
  async rateConversation(id: string, rating: number, comment?: string) {
    return this.request(`/api/conversations/${id}/rate`, {
      method: "POST",
      body: JSON.stringify({ rating, comment }),
    });
  }

  /** 知识库搜索 */
  async searchKnowledge(query: string, topK = 5) {
    return this.request(`/api/knowledge/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
  }

  /** 系统统计 */
  async getStats() {
    return this.request<Record<string, unknown>>("/api/admin/stats");
  }

  /** 对话日志 */
  async getLogs(page = 1, limit = 20) {
    return this.request(`/api/admin/logs?page=${page}&limit=${limit}`);
  }

  /** 解析 SSE 流 */
  static parseSSEStream(
    stream: ReadableStream<Uint8Array>,
    onToken: (token: string) => void,
    onEvent: (event: SSETokenEvent) => void,
    onError: (err: Error) => void,
    onDone: () => void
  ): AbortController {
    const controller = new AbortController();
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (controller.signal.aborted) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          let currentEvent = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            }
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                // token: backend sends {"content": "..."} without type field
                if (data.content) {
                  onToken(data.content);
                }
                // metadata
                if (currentEvent === "metadata") {
                  onEvent({ type: "metadata", data } as SSETokenEvent);
                }
                // error
                if (currentEvent === "error") {
                  onError(new Error(data.error || "Stream error"));
                  onDone();
                  return;
                }
                // done
                if (currentEvent === "done") {
                  onDone();
                  return;
                }
              } catch {
                // skip non-JSON lines
              }
              currentEvent = "";
            }
          }
        }
        onDone();
      } catch (err) {
        if (!controller.signal.aborted) {
          onError(err instanceof Error ? err : new Error(String(err)));
          onDone();
        }
      }
    })();

    return controller;
  }
}

export { ApiClient };
export const apiClient = new ApiClient(API_BASE);
export default apiClient;
