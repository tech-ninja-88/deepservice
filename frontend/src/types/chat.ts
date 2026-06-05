export interface Conversation {
  id: string;
  title: string;
  status: "active" | "waiting" | "transferred" | "closed";
  message_count: number;
  created_at: string;
  updated_at: string;
  last_message?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system" | "human_agent";
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  intent?: string;
  confidence?: number;
  sources?: Citation[];
  tokens?: { prompt_tokens: number; completion_tokens: number };
  hallucination_risk?: number;
}

export interface Citation {
  index: number;
  title: string;
  snippet: string;
  score: number;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  stream?: boolean;
}

export interface ChatResponse {
  conversation_id: string;
  content: string;
  response_type: string;
  confidence: number;
  metadata?: Record<string, unknown>;
  sources?: { index: number; title: string; score: number }[];
  elapsed_seconds?: number;
  timestamp?: string;
}

export interface SSETokenEvent {
  type: "token" | "metadata" | "done" | "error" | "intent" | "blocked";
  content?: string;
  data?: Record<string, unknown>;
  error?: string;
}

export interface ConversationListItem {
  id: string;
  title: string;
  last_message: string;
  updated_at: string;
  message_count: number;
}
