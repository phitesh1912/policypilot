import { useEffect, useRef, useState } from "react";
import {
  ask,
  askStream,
  getHealth,
  type AskResult,
  type Health,
  type Region,
  type RetrievalTrace,
  type Role,
} from "./api";

const SAMPLES = [
  "How many days of casual leave do I get per year?",
  "What is the overtime rate and who is eligible?",
  "How much is the wellness allowance?",
  "Can I work from another country for a month?",
  "How much does Acme match on the 401(k)?",
  "What is the office wifi password?",
];

const CITATION_RE = /\[([a-z0-9-]+):([a-z0-9-]+)\]/g;

function AnswerText({ text }: { text: string }) {
  const parts: Array<string | { tag: string }> = [];
  let last = 0;
  for (const m of text.matchAll(CITATION_RE)) {
    parts.push(text.slice(last, m.index));
    parts.push({ tag: m[0] });
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(text.slice(last));
  return (
    <p className="answer-text">
      {parts.map((p, i) =>
        typeof p === "string" ? <span key={i}>{p}</span> : <cite key={i}>{p.tag}</cite>,
      )}
    </p>
  );
}

export default function App() {
  const [question, setQuestion] = useState(SAMPLES[0]);
  const [role, setRole] = useState<Role>("employee");
  const [region, setRegion] = useState<Region>("IN");
  const [streaming, setStreaming] = useState(true);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [trace, setTrace] = useState<RetrievalTrace[]>([]);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, [result]);

  async function submit() {
    if (!question.trim() || busy) return;
    cancelRef.current?.();
    setBusy(true);
    setError(null);
    setDraft("");
    setTrace([]);
    setResult(null);
    if (streaming) {
      cancelRef.current = askStream(question, role, region, {
        onMeta: setTrace,
        onToken: (t) => setDraft((d) => d + t),
        onDone: (r) => {
          setResult(r);
          setBusy(false);
        },
        onError: (e) => {
          setError(e.message);
          setBusy(false);
        },
      });
    } else {
      try {
        const r = await ask(question, role, region);
        setResult(r);
        setTrace(r.retrieved);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    }
  }

  const shown = result ? result.answer : draft;

  return (
    <div className="shell">
      <header>
        <h1>PolicyPilot</h1>
        <p className="tagline">
          HR policy &amp; payroll answers, grounded in the Acme handbook with citations — or an
          honest “I don’t know”.
        </p>
        {health && (
          <p className="health">
            {health.provider} · {health.chat_model} · {health.retriever} retrieval ·{" "}
            {health.indexed_chunks} chunks · cache {health.cache.hits}/{health.cache.hits +
              health.cache.misses}{" "}
            hits
          </p>
        )}
      </header>

      <section className="ask">
        <div className="identity">
          <label>
            Role
            <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
              <option value="employee">employee</option>
              <option value="manager">manager</option>
              <option value="hr">hr</option>
              <option value="admin">admin</option>
            </select>
          </label>
          <label>
            Region
            <select value={region} onChange={(e) => setRegion(e.target.value as Region)}>
              <option value="IN">IN</option>
              <option value="US">US</option>
              <option value="GLOBAL">GLOBAL</option>
            </select>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
            />
            stream
          </label>
        </div>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          maxLength={2000}
          placeholder="Ask about leave, pay, benefits, remote work…"
        />
        <div className="row">
          <div className="samples">
            {SAMPLES.map((s) => (
              <button key={s} className="chip" onClick={() => setQuestion(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
          <button className="primary" onClick={submit} disabled={busy || !question.trim()}>
            {busy ? "Thinking…" : "Ask"}
          </button>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {(shown || busy) && (
        <section className={`answer ${result?.refused ? "refused" : ""}`}>
          <div className="answer-head">
            <h2>Answer</h2>
            {result && (
              <div className="badges">
                {result.cached && <span className="badge">cached</span>}
                {result.refused && <span className="badge warn">refused</span>}
                <span className="badge">{result.latency_ms} ms</span>
              </div>
            )}
          </div>
          <AnswerText text={shown || "…"} />
          {result && result.citations.length > 0 && (
            <ul className="citations">
              {result.citations.map((c) => (
                <li key={c.tag}>
                  <cite>{c.tag}</cite> {c.title} — {c.section}{" "}
                  <span className="muted">
                    ({c.region}, v{c.version})
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {result && (
        <section className="usage">
          <h2>Tokens &amp; cost</h2>
          <dl>
            <div>
              <dt>context</dt>
              <dd>{result.tokens.context}</dd>
            </div>
            <div>
              <dt>prompt</dt>
              <dd>{result.tokens.prompt}</dd>
            </div>
            <div>
              <dt>completion</dt>
              <dd>{result.tokens.completion}</dd>
            </div>
            <div>
              <dt>embedding</dt>
              <dd>{result.tokens.embedding}</dd>
            </div>
            <div>
              <dt>est. cost</dt>
              <dd>${result.cost_usd.toFixed(6)}</dd>
            </div>
          </dl>
        </section>
      )}

      {trace.length > 0 && (
        <section className="trace">
          <h2>Retrieval trace</h2>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>document</th>
                <th>section</th>
                <th>fused</th>
                <th>bm25</th>
                <th>vector</th>
                <th>in ctx</th>
              </tr>
            </thead>
            <tbody>
              {trace.map((t, i) => (
                <tr key={t.chunk_id} className={t.in_context ? "" : "dim"}>
                  <td>{i + 1}</td>
                  <td>{t.doc_id}</td>
                  <td>{t.section}</td>
                  <td>{t.score.toFixed(4)}</td>
                  <td>{t.sources.bm25 ?? "–"}</td>
                  <td>{t.sources.vector ?? "–"}</td>
                  <td>{t.in_context ? "✓" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <footer>
        Synthetic Acme Corp handbook. Identity is chosen here for demo purposes; in production it
        comes from the auth token.
      </footer>
    </div>
  );
}
