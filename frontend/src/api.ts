export type Role = "employee" | "manager" | "hr" | "admin";
export type Region = "IN" | "US" | "GLOBAL";

export interface Citation {
  tag: string;
  doc_id: string;
  title: string;
  section: string;
  chunk_id: string;
  region: Region;
  version: string;
}

export interface TokenUsage {
  prompt: number;
  completion: number;
  embedding: number;
  context: number;
}

export interface RetrievalTrace {
  chunk_id: string;
  doc_id: string;
  section: string;
  score: number;
  sources: Record<string, number>;
  in_context: boolean;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  refused: boolean;
  cached: boolean;
  tokens: TokenUsage;
  cost_usd: number;
  retrieved: RetrievalTrace[];
  latency_ms: number;
}

export interface Health {
  status: string;
  version: string;
  provider: string;
  chat_model: string;
  embed_model: string;
  retriever: string;
  indexed_chunks: number;
  cache: { size: number; hits: number; misses: number };
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function ask(question: string, role: Role, region: Region): Promise<AskResult> {
  const r = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, user: { role, region } }),
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`ask ${r.status}: ${detail}`);
  }
  return r.json();
}

export interface StreamHandlers {
  onMeta: (retrieved: RetrievalTrace[]) => void;
  onToken: (text: string) => void;
  onDone: (result: AskResult) => void;
  onError: (err: Error) => void;
}

export function askStream(
  question: string,
  role: Role,
  region: Region,
  h: StreamHandlers,
): () => void {
  const params = new URLSearchParams({ question, role, region });
  const es = new EventSource(`/ask/stream?${params}`);
  es.addEventListener("meta", (e) => h.onMeta(JSON.parse((e as MessageEvent).data).retrieved));
  es.addEventListener("token", (e) => h.onToken(JSON.parse((e as MessageEvent).data).text));
  es.addEventListener("done", (e) => {
    h.onDone(JSON.parse((e as MessageEvent).data).result);
    es.close();
  });
  es.onerror = () => {
    // EventSource fires onerror on normal close too; only report if we never got "done".
    if (es.readyState !== EventSource.CLOSED) h.onError(new Error("stream error"));
    es.close();
  };
  return () => es.close();
}
