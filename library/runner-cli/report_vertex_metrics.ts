// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0
//
// Reads vertex_requests.jsonl and generates an HTML dashboard with charts.

import * as fs from "fs";
import { readJsonl, pairLogs, bucketize, overall } from "../src/metrics/aggregate";

export async function generateDashboard(jsonlPath: string, outHtmlPath: string) {
  const logs = readJsonl(jsonlPath);
  const combined = pairLogs(logs);
  const buckets = bucketize(combined, 1000); // 1s buckets
  const totals = overall(combined);

  const seriesTime = buckets.map((b) => ({ t: b.t, total: b.total, ok: b.ok, error: b.error }));
  const seriesLatency = buckets.map((b) => ({ t: b.t, p50: b.p50, p90: b.p90, p99: b.p99 }));

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Vertex Load Test Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .cards { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-width: 200px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 24px; }
    canvas { background: #fff; border: 1px solid #eee; }
    .muted { color: #666; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
  <h2>Vertex Load Test Dashboard</h2>
  <div class="cards">
    <div class="card"><div>Total requests</div><div style="font-size:28px">${totals.total}</div></div>
    <div class="card"><div>Success</div><div style="font-size:28px;color:#2e7d32">${totals.ok}</div></div>
    <div class="card"><div>Errors</div><div style="font-size:28px;color:#c62828">${totals.error}</div></div>
    <div class="card"><div>Latency p50</div><div style="font-size:28px">${Math.round(totals.p50)} ms</div></div>
    <div class="card"><div>Latency p90</div><div style="font-size:28px">${Math.round(totals.p90)} ms</div></div>
    <div class="card"><div>Latency p99</div><div style="font-size:28px">${Math.round(totals.p99)} ms</div></div>
  </div>

  <div class="grid">
    <div>
      <h3>Requests over time</h3>
      <canvas id="reqChart" height="120"></canvas>
      <div class="muted">Counts per 1s bucket</div>
    </div>
    <div>
      <h3>Latency over time</h3>
      <canvas id="latChart" height="120"></canvas>
      <div class="muted">p50/p90/p99 per 1s bucket</div>
    </div>
  </div>

<script>
const seriesTime = ${JSON.stringify(seriesTime)};
const seriesLatency = ${JSON.stringify(seriesLatency)};

function fmtTs(ms){ const d=new Date(ms); return d.toLocaleTimeString(); }

function toDatasetTime(key, label, color){
  return {
    label,
    data: seriesTime.map(p => ({ x: p.t, y: p[key] })),
    borderColor: color,
    backgroundColor: color,
    tension: 0.2,
  };
}

function toDatasetLat(key, label, color){
  return {
    label,
    data: seriesLatency.map(p => ({ x: p.t, y: p[key] })),
    borderColor: color,
    backgroundColor: color,
    tension: 0.2,
  };
}

const ctx1 = document.getElementById('reqChart');
new Chart(ctx1, {
  type: 'line',
  data: {
    datasets: [
      toDatasetTime('total', 'Total', '#1565c0'),
      toDatasetTime('ok', 'OK', '#2e7d32'),
      toDatasetTime('error', 'Error', '#c62828'),
    ]
  },
  options: {
    parsing: false,
    scales: {
      x: { type: 'timeseries', adapters: { date: {} }, ticks: { callback: v => fmtTs(v) } },
      y: { beginAtZero: true }
    }
  }
});

// Fallback to linear scale for time to avoid external date adapters
// We map x values (epoch ms) directly and format labels via fmtTs
const ctx2 = document.getElementById('latChart');
new Chart(ctx2, {
  type: 'line',
  data: {
    datasets: [
      toDatasetLat('p50', 'p50', '#00796b'),
      toDatasetLat('p90', 'p90', '#f9a825'),
      toDatasetLat('p99', 'p99', '#6a1b9a'),
    ]
  },
  options: {
    parsing: false,
    scales: {
      x: { type: 'timeseries', adapters: { date: {} }, ticks: { callback: v => fmtTs(v) } },
      y: { beginAtZero: true }
    }
  }
});
</script>
</body>
</html>`;

  fs.writeFileSync(outHtmlPath, html, "utf-8");
}

if (require.main === module) {
  const jsonl = process.argv[2];
  const out = process.argv[3] || jsonl.replace(/\.jsonl$/, ".html");
  generateDashboard(jsonl, out).catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
