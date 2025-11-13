// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0
// See LICENSE

import * as fs from "fs";
import * as path from "path";

export type RequestLog = {
  id: string;
  tsStart: number; // epoch ms
  tsEnd?: number; // epoch ms
  op?: string; // e.g., categorize, summarize, learnTopics
  userId?: string | number;
  status?: number | string; // http status code or grpc code
  ok?: boolean;
  latencyMs?: number;
  errorMessage?: string;
};

function getErrorMessage(err: unknown): string | undefined {
  if (!err) return undefined;
  if (typeof err === "string") return err;
  if (typeof err === "object") {
    const anyErr = err as { message?: unknown };
    const m = anyErr.message;
    if (typeof m === "string") return m;
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }
  return String(err);
}

/**
 * Lightweight JSONL logger + basic aggregator for Vertex requests.
 * This is a singleton; initialize once via init(outDir) in CLI harnesses.
 */
class VertexMetricsImpl {
  private stream: fs.WriteStream | null = null;
  private nextId = 1;
  private enabled = false;
  private currentUser: string | null = null;
  private startTimes: Map<string, number> = new Map();

  init(outDir: string) {
    fs.mkdirSync(outDir, { recursive: true });
    const filePath = path.join(outDir, "vertex_requests.jsonl");
    this.stream = fs.createWriteStream(filePath, { flags: "a" });
    this.enabled = true;
  }

  setCurrentUser(userId: string | number | null) {
    this.currentUser = userId === null ? null : String(userId);
  }

  isEnabled() {
    return this.enabled && !!this.stream;
  }

  start(op?: string, userId?: string | number): string {
    if (!this.isEnabled()) return "";
    const id = String(this.nextId++);
    const uid = userId != null ? String(userId) : (this.currentUser ?? undefined);
    const tsStart = Date.now();
    this.startTimes.set(id, tsStart);
    const rec: RequestLog = { id, tsStart, op, userId: uid };
    this.write(rec);
    return id;
  }

  end(id: string, ok: boolean, status?: number | string, error?: unknown) {
    if (!this.isEnabled() || !id) return;
    const tsEnd = Date.now();
    const tsStart = this.startTimes.get(id) ?? tsEnd; // fallback to satisfy typing
    const latencyMs = Math.max(0, tsEnd - tsStart);
    this.startTimes.delete(id);
    const rec: RequestLog = {
      id,
      tsStart,
      tsEnd,
      ok,
      status,
      latencyMs,
      errorMessage: getErrorMessage(error),
    };
    this.write(rec);
  }

  private write(obj: RequestLog) {
    if (!this.stream) return;
    try {
      this.stream.write(JSON.stringify(obj) + "\n");
    } catch {
      // swallow
    }
  }

  close() {
    if (this.stream) this.stream.end();
    this.stream = null;
    this.enabled = false;
  }
}

export const VertexMetrics = new VertexMetricsImpl();
