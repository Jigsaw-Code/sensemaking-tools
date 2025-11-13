// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0

import { GenerativeModel } from "@google-cloud/vertexai";
import { VertexModel } from "./vertex_model";
import { VertexMetrics } from "../metrics/vertex_metrics";

/**
 * A VertexModel wrapper that logs per-request start/end to VertexMetrics.
 */
export class MetricsVertexModel extends VertexModel {
  private userId?: string;

  constructor(project: string, location: string, modelName?: string, userId?: string | number) {
    super(project, location, modelName);
    this.userId = userId != null ? String(userId) : undefined;
  }

  setUserId(userId?: string | number) {
    this.userId = userId != null ? String(userId) : undefined;
  }

  override async callLLM(
    prompt: string,
    model: GenerativeModel,
    validator: (response: string) => boolean = () => true
  ): Promise<string> {
    const reqId = VertexMetrics.start("callLLM", this.userId);
    try {
      const res = await super.callLLM(prompt, model, validator);
      VertexMetrics.end(reqId, true, 200);
      return res;
    } catch (e: unknown) {
      // Try to extract a status/code without relying on 'any'
      const status =
        typeof e === "object" && e !== null && (e as { status?: unknown; code?: unknown }).status
          ? (e as { status?: number | string }).status
          : typeof e === "object" && e !== null && (e as { code?: unknown }).code
            ? (e as { code?: number | string }).code
            : 500;
      VertexMetrics.end(reqId, false, status, e);
      throw e;
    }
  }
}
