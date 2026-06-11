import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar,
  Cell, PieChart, Pie, Legend
} from "recharts";

// ── Real backtest data from NEXUS ──────────────────────────────────────────
const EQUITY = [{"t":"2022-09-25","v":1000},{"t":"2022-10-19","v":979.5},{"t":"2022-11-13","v":966},{"t":"2022-12-08","v":974.36},{"t":"2023-01-02","v":964.21},{"t":"2023-01-27","v":945.92},{"t":"2023-02-21","v":944.32},{"t":"2023-03-18","v":1001.02},{"t":"2023-04-12","v":980.52},{"t":"2023-05-07","v":980.51},{"t":"2023-06-01","v":999.75},{"t":"2023-06-26","v":979.95},{"t":"2023-07-21","v":959.6},{"t":"2023-08-15","v":950.95},{"t":"2023-09-09","v":943.02},{"t":"2023-10-04","v":962.73},{"t":"2023-10-29","v":969.25},{"t":"2023-11-23","v":949.07},{"t":"2023-12-18","v":919.59},{"t":"2024-01-12","v":901.73},{"t":"2024-02-06","v":919.49},{"t":"2024-03-02","v":940.45},{"t":"2024-03-27","v":997.05},{"t":"2024-04-21","v":998.51},{"t":"2024-05-16","v":988.28},{"t":"2024-06-10","v":1012.23},{"t":"2024-07-05","v":970.72},{"t":"2024-07-30","v":989.8},{"t":"2024-08-24","v":988.67},{"t":"2024-09-18","v":1008.11},{"t":"2024-10-13","v":987.37},{"t":"2024-11-07","v":1015.71},{"t":"2024-12-02","v":964.1},{"t":"2024-12-27","v":980.69},{"t":"2025-01-21","v":999.99},{"t":"2025-02-15","v":989.68},{"t":"2025-03-12","v":949.51},{"t":"2025-04-06","v":954.32},{"t":"2025-05-01","v":992.28},{"t":"2025-05-26","v":1031.5},{"t":"2025-06-20","v":1072.17},{"t":"2025-07-15","v":1133.56},{"t":"2025-08-09","v":1123.26},{"t":"2025-09-03","v":1134.13},{"t":"2025-09-28","v":1113},{"t":"2025-10-23","v":1099.84},{"t":"2025-11-17","v":1077.19},{"t":"2025-12-12","v":1106.89},{"t":"2026-01-06","v":1099.66},{"t":"2026-01-31","v":1065.54},{"t":"2026-02-25","v":1043.67},{"t":"2026-03-22","v":1055.76},{"t":"2026-04-16","v":1050.95},{"t":"2026-05-11","v":1060.35},{"t":"2026-06-05","v":1078.62}];
const TRADES = [{"entry":"2026-01-06","dir":"SHORT","entry_p":93876.95,"exit_p":97615.98,"pnl":-11.5,"reason":"SL","conf":0.664},{"entry":"2026-01-21","dir":"LONG","entry_p":88326.51,"exit_p":84629.1,"pnl":-11.4,"reason":"SL","conf":0.565},{"entry":"2026-01-30","dir":"LONG","entry_p":84562.73,"exit_p":80600.0,"pnl":-11.23,"reason":"SL","conf":0.636},{"entry":"2026-02-01","dir":"LONG","entry_p":78626.12,"exit_p":73993.17,"pnl":-11.02,"reason":"SL","conf":0.746},{"entry":"2026-02-04","dir":"LONG","entry_p":75640.09,"exit_p":70332.03,"pnl":-10.85,"reason":"SL","conf":0.876},{"entry":"2026-02-06","dir":"LONG","entry_p":62704.45,"exit_p":67453.77,"pnl":7.0,"reason":"MAX_HOLD","conf":0.843},{"entry":"2026-02-27","dir":"LONG","entry_p":67456.52,"exit_p":69912.79,"pnl":5.09,"reason":"MAX_HOLD","conf":0.566},{"entry":"2026-03-20","dir":"SHORT","entry_p":69911.53,"exit_p":71767.83,"pnl":-4.81,"reason":"MAX_HOLD","conf":0.584},{"entry":"2026-04-10","dir":"LONG","entry_p":71774.37,"exit_p":79268.11,"pnl":20.62,"reason":"TP","conf":0.569},{"entry":"2026-04-26","dir":"SHORT","entry_p":77613.12,"exit_p":80949.3,"pnl":-11.21,"reason":"SL","conf":0.593},{"entry":"2026-05-07","dir":"SHORT","entry_p":81428.85,"exit_p":75062.92,"pnl":20.19,"reason":"TP","conf":0.746},{"entry":"2026-05-29","dir":"SHORT","entry_p":73537.03,"exit_p":67726.92,"pnl":20.8,"reason":"TP","conf":0.6},{"entry":"2026-06-03","dir":"LONG","entry_p":66694.01,"exit_p":63352.19,"pnl":-11.45,"reason":"SL","conf":0.799},{"entry":"2026-06-05","dir":"LONG","entry_p":63807.69,"exit_p":60081.57,"pnl":-11.27,"reason":"SL","conf":0.782},{"entry":"2026-06-06","dir":"LONG","entry_p":60924.48,"exit_p":61449.29,"pnl":1.18,"reason":"END","conf":0.882}];
const FEATURES = [{"name":"swing_low","val":1.816},{"name":"day_of_week","val":1.814},{"name":"vol_ma_10","val":1.415},{"name":"dow_sin","val":1.386},{"name":"rsi_7","val":1.353},{"name":"rsi_14_lag3","val":1.341},{"name":"rel_vol_lag5","val":1.314},{"name":"hl_range","val":1.302},{"name":"rsi_14","val":1.297},{"name":"bb_pct","val":1.281}];
const EXIT_PIE = [{ name: "Stop Loss", value: 67, color: "#ef4444" },{ name: "Max Hold", value: 18, color: "#f59e0b" },{ name: "Take Profit", value: 36, color: "#10b981" },{ name: "End", value: 1, color: "#6b7280" }];

const s = {
  green: "#10b981", red: "#ef4444", cyan: "#06b6d4",
  amber: "#f59e0b", muted: "var(--color-text-secondary)",
  border: "var(--color-border-tertiary)",
  card: "var(--color-background-secondary)",
  text: "var(--color-text-primary)",
};

const fmt = (n, dec = 2) => n.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
const fmtK = (n) => n >= 1000 ? `$${fmt(n / 1000)}K` : `$${fmt(n)}`;

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, color: s.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600, color: color || s.text, fontFamily: "var(--font-mono, monospace)", lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: s.muted, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function SignalBar({ label, pct, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
        <span style={{ color: s.muted }}>{label}</span>
        <span style={{ fontFamily: "monospace", fontWeight: 500, color }}>{(pct * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: s.border }}>
        <div style={{ height: "100%", borderRadius: 3, background: color, width: `${pct * 100}%`, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

export default function NEXUSDashboard() {
  const [tab, setTab] = useState("journal");
  const totalReturn = 7.98;
  const capital = 1079.81;

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0];
    return (
      <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 8, padding: "8px 12px", fontSize: 12 }}>
        <div style={{ color: s.muted }}>{d.payload.t}</div>
        <div style={{ color: d.value >= 1000 ? s.green : s.red, fontFamily: "monospace", fontWeight: 600 }}>${fmt(d.value)}</div>
      </div>
    );
  };

  return (
    <div style={{ padding: "0 0 24px", fontFamily: "var(--font-sans)", color: s.text }}>

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 0 12px", borderBottom: `1px solid ${s.border}`, marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, background: s.cyan, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ color: "#fff", fontWeight: 700, fontSize: 13 }}>N</span>
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>NEXUS</div>
            <div style={{ fontSize: 11, color: s.muted }}>AI Trading System</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 11, background: "#fef3c7", color: "#92400e", borderRadius: 999, padding: "3px 10px", fontWeight: 500 }}>● PAPER MODE</span>
          <span style={{ fontSize: 11, background: "#d1fae5", color: "#065f46", borderRadius: 999, padding: "3px 10px", fontWeight: 500 }}>5Y BTC Backtest</span>
        </div>
      </div>

      {/* ── Top metrics ────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <MetricCard label="Final Capital" value={`$${fmt(capital)}`} sub="Started $1,000" color={s.green} />
        <MetricCard label="Total Return" value={`+${totalReturn}%`} sub="3.7 years" color={s.green} />
        <MetricCard label="Win Rate" value="41.8%" sub="51 / 122 trades" />
        <MetricCard label="Profit Factor" value="1.11×" sub="Needs 1.5×+ for live" color={s.amber} />
        <MetricCard label="Max Drawdown" value="-9.92%" sub="Jan 2024 trough" color={s.red} />
        <MetricCard label="Sharpe Ratio" value="0.256" sub="Target: > 1.0" color={s.amber} />
      </div>

      {/* ── Equity Curve ───────────────────────────────────────────── */}
      <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: "14px 12px 8px", marginBottom: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 10, color: s.muted }}>Equity curve — out-of-sample backtest (2022–2026)</div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={EQUITY} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={s.border} />
            <XAxis dataKey="t" tick={{ fontSize: 9, fill: s.muted }} tickFormatter={v => v.slice(0, 7)} interval={8} />
            <YAxis tick={{ fontSize: 9, fill: s.muted }} domain={["auto", "auto"]} tickFormatter={v => `$${v}`} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={1000} stroke={s.muted} strokeDasharray="4 4" strokeWidth={1} />
            <Line type="monotone" dataKey="v" stroke={s.cyan} strokeWidth={2} dot={false} activeDot={{ r: 3, fill: s.cyan }} />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 10, color: s.muted, textAlign: "center", marginTop: 2 }}>Dashed line = starting capital $1,000</div>
      </div>

      {/* ── Signal + Features row ──────────────────────────────────── */}
      <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>

        {/* Live signal */}
        <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: 14, flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: s.muted, marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>Latest signal</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
            <div style={{ background: "#d1fae5", color: "#065f46", borderRadius: 8, padding: "6px 14px", fontWeight: 700, fontSize: 16 }}>BUY</div>
            <div style={{ fontSize: 11, color: s.muted }}>model output<br />sentiment: HOLD ⚠️</div>
          </div>
          <SignalBar label="BUY probability" pct={0.6174} color={s.green} />
          <SignalBar label="SELL probability" pct={0.298} color={s.red} />
          <SignalBar label="HOLD probability" pct={0.0846} color={s.muted} />
          <div style={{ marginTop: 10, padding: "8px 10px", background: "#fef3c7", borderRadius: 8, fontSize: 11 }}>
            <span style={{ color: "#92400e", fontWeight: 500 }}>⚠️ Signal Engine → HOLD</span>
            <div style={{ color: "#92400e", marginTop: 2 }}>Fear & Greed: 12 (Extreme Fear) — sentiment vetoed BUY</div>
          </div>
          <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", fontSize: 11, color: s.muted }}>
            <span>Sentiment score: -0.27</span>
            <span>Confidence: 31%</span>
          </div>
        </div>

        {/* Feature importance */}
        <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: 14, flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: s.muted, marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>Top 10 features</div>
          {FEATURES.map((f, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                <span style={{ color: s.muted, fontFamily: "monospace" }}>{f.name}</span>
                <span style={{ color: s.text, fontWeight: 500 }}>{f.val}%</span>
              </div>
              <div style={{ height: 4, borderRadius: 2, background: s.border }}>
                <div style={{ height: "100%", borderRadius: 2, background: s.cyan, width: `${(f.val / 1.816) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Model metrics + Exit distribution ──────────────────────── */}
      <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>

        {/* Model metrics */}
        <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: 14, flex: 1.2 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: s.muted, marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>XGBoost model (walk-forward)</div>
          {[
            { label: "Overall accuracy", value: "45.1%", note: "vs 33.3% random baseline", ok: true },
            { label: "Weighted F1", value: "0.419", note: "across BUY/SELL/HOLD", ok: true },
            { label: "BUY signal F1", value: "0.479", note: "precision on upside calls", ok: true },
            { label: "SELL signal F1", value: "0.500", note: "precision on downside calls", ok: true },
            { label: "Training data", value: "1,626 bars", note: "5 years BTC daily", ok: true },
            { label: "Validation splits", value: "5 folds", note: "walk-forward, no lookahead", ok: true },
          ].map((r, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "5px 0", borderBottom: i < 5 ? `1px solid ${s.border}` : "none", fontSize: 12 }}>
              <span style={{ color: s.muted }}>{r.label}</span>
              <div style={{ textAlign: "right" }}>
                <span style={{ fontFamily: "monospace", fontWeight: 500, color: r.ok ? s.green : s.red }}>{r.value}</span>
                <div style={{ fontSize: 10, color: s.muted }}>{r.note}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Exit distribution */}
        <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: 14, flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: s.muted, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Exit distribution (122 trades)</div>
          <ResponsiveContainer width="100%" height={130}>
            <PieChart>
              <Pie data={EXIT_PIE} cx="50%" cy="50%" innerRadius={35} outerRadius={55} paddingAngle={2} dataKey="value">
                {EXIT_PIE.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip formatter={(v, n) => [v, n]} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px", marginTop: 4 }}>
            {EXIT_PIE.map((e, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: e.color, flexShrink: 0 }} />
                <span style={{ color: s.muted }}>{e.name}: <strong style={{ color: s.text }}>{e.value}</strong></span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: s.muted, padding: "6px 8px", background: "#fef3c7", borderRadius: 6 }}>
            ⚠️ SL exits (67) vs TP exits (36) — filter to conf &gt; 0.65 to improve
          </div>
        </div>
      </div>

      {/* ── Trade Journal ──────────────────────────────────────────── */}
      <div style={{ background: s.card, border: `1px solid ${s.border}`, borderRadius: 12, padding: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: s.muted, textTransform: "uppercase", letterSpacing: "0.06em" }}>Recent trades (last 15 of 122)</div>
          <div style={{ fontSize: 11 }}>
            <span style={{ color: s.green }}>✓ 51 wins</span>
            <span style={{ color: s.muted }}> / </span>
            <span style={{ color: s.red }}>✗ 71 losses</span>
          </div>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "monospace" }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${s.border}` }}>
                {["Date", "Dir", "Entry", "Exit", "P&L", "Reason", "Conf"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "4px 8px", color: s.muted, fontWeight: 500, fontFamily: "var(--font-sans)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TRADES.map((t, i) => {
                const win = t.pnl > 0;
                const rowColor = win ? { background: "rgba(16,185,129,0.04)" } : {};
                return (
                  <tr key={i} style={{ borderBottom: `1px solid ${s.border}`, ...rowColor }}>
                    <td style={{ padding: "5px 8px", color: s.muted }}>{t.entry}</td>
                    <td style={{ padding: "5px 8px" }}>
                      <span style={{ color: t.dir === "LONG" ? s.green : s.red, fontWeight: 600 }}>{t.dir}</span>
                    </td>
                    <td style={{ padding: "5px 8px", color: s.text }}>${(t.entry_p / 1000).toFixed(1)}K</td>
                    <td style={{ padding: "5px 8px", color: s.text }}>${(t.exit_p / 1000).toFixed(1)}K</td>
                    <td style={{ padding: "5px 8px", fontWeight: 600, color: win ? s.green : s.red }}>
                      {win ? "+" : ""}{t.pnl.toFixed(2)}
                    </td>
                    <td style={{ padding: "5px 8px" }}>
                      <span style={{
                        fontSize: 10, borderRadius: 4, padding: "1px 6px",
                        background: t.reason === "TP" ? "#d1fae5" : t.reason === "SL" ? "#fee2e2" : "#fef3c7",
                        color: t.reason === "TP" ? "#065f46" : t.reason === "SL" ? "#991b1b" : "#92400e"
                      }}>{t.reason}</span>
                    </td>
                    <td style={{ padding: "5px 8px", color: s.muted }}>{(t.conf * 100).toFixed(0)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Improvement checklist ──────────────────────────────────── */}
      <div style={{ marginTop: 14, padding: "12px 14px", border: `1px solid ${s.border}`, borderRadius: 12, background: s.card }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>Phase 2 improvements to hit 1.5× profit factor</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 11 }}>
          {[
            { done: false, text: "Switch to hourly BTC/USDT (less noise)" },
            { done: false, text: "Raise min_confidence to 0.65+" },
            { done: true,  text: "Walk-forward validation ✓" },
            { done: true,  text: "102 features engineered ✓" },
            { done: false, text: "Add news sentiment to model features" },
            { done: false, text: "Tune ATR multiplier (try 2.0)" },
            { done: true,  text: "Risk 1% max per trade ✓" },
            { done: false, text: "Add regime filter (no trades in sideways)" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start" }}>
              <span style={{ color: item.done ? s.green : s.muted, flexShrink: 0 }}>{item.done ? "✓" : "○"}</span>
              <span style={{ color: item.done ? s.text : s.muted }}>{item.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
