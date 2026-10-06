/**
 * @fileoverview Tests for palette.js.
 * Run with `node --test` from src/report_ui.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEFAULT_DEMOGRAPHIC_BASE_COLOR,
  DEMOGRAPHIC_PALETTE_STEPS,
  LIGHTEST_CHROMA_FRACTION,
  PALETTE_MAX_LIGHTNESS,
  PALETTE_MIN_LIGHTNESS,
  generateSequentialPalette,
  isHexColor,
  resolveDemographicColors,
} from "./palette.js";

/**
 * Extracts the base color and target lightness from a generated
 * `oklch(from <base> <targetL> calc(...) h)` string.
 * @param {string} cssColor
 * @returns {{ base: string, targetL: number }}
 */
function parseOklchStep(cssColor) {
  const match =
    /^oklch\(from (#[0-9a-f]{3,6}) ([0-9.]+) calc\(c \* min\(.+\)\) h\)$/.exec(
      cssColor,
    );
  assert.ok(match, `Unexpected CSS color format: ${cssColor}`);
  return { base: match[1], targetL: Number(match[2]) };
}

describe("isHexColor", () => {
  it("accepts 6- and 3-digit hex strings, case-insensitively", () => {
    assert.equal(isHexColor("#ff0000"), true);
    assert.equal(isHexColor("#F00"), true);
    assert.equal(isHexColor(" #00ff00 "), true);
  });

  it("rejects non-hex values", () => {
    for (const bad of ["red", "#ff00", "ff0000", "#gggggg", "", null, 5]) {
      assert.equal(isHexColor(bad), false, String(bad));
    }
  });
});

describe("generateSequentialPalette", () => {
  it("returns six CSS oklch(from ...) strings by default", () => {
    const palette = generateSequentialPalette("#4886f7");
    assert.equal(palette.length, DEMOGRAPHIC_PALETTE_STEPS);
    for (const color of palette) {
      const { base } = parseOklchStep(color);
      assert.equal(base, "#4886f7");
    }
  });

  it("normalizes base hex casing and whitespace", () => {
    const palette = generateSequentialPalette("  #DA3D2E ");
    for (const color of palette) {
      assert.equal(parseOklchStep(color).base, "#da3d2e");
    }
  });

  it("spaces lightness evenly from PALETTE_MAX_LIGHTNESS to PALETTE_MIN_LIGHTNESS", () => {
    const ls = generateSequentialPalette("#00885F").map(
      (c) => parseOklchStep(c).targetL,
    );
    assert.equal(ls[0], PALETTE_MAX_LIGHTNESS);
    assert.equal(ls.at(-1), PALETTE_MIN_LIGHTNESS);
    for (let i = 1; i < ls.length; i++) {
      assert.ok(ls[i] < ls[i - 1], `step ${i}: ${ls}`);
      const gap = ls[i - 1] - ls[i];
      const expectedGap =
        (PALETTE_MAX_LIGHTNESS - PALETTE_MIN_LIGHTNESS) /
        (DEMOGRAPHIC_PALETTE_STEPS - 1);
      assert.ok(Math.abs(gap - expectedGap) < 1e-4, `gap ${gap}`);
    }
  });

  it("includes the chroma taper factor and zero-division guards in calc()", () => {
    const [lightest] = generateSequentialPalette("#4886f7");
    assert.ok(lightest.includes(`1 - ${1 - LIGHTEST_CHROMA_FRACTION} *`));
    assert.ok(lightest.includes("max(l, 0.001)"));
    assert.ok(lightest.includes(`max(0.001, ${PALETTE_MAX_LIGHTNESS} - l)`));
  });

  it("supports custom step counts", () => {
    const two = generateSequentialPalette("#4886f7", 2).map(
      (c) => parseOklchStep(c).targetL,
    );
    assert.deepEqual(two, [PALETTE_MAX_LIGHTNESS, PALETTE_MIN_LIGHTNESS]);
    assert.equal(generateSequentialPalette("#4886f7", 9).length, 9);
  });

  it("throws on invalid input", () => {
    assert.throws(() => generateSequentialPalette("blue"), /Invalid hex/);
    assert.throws(() => generateSequentialPalette("#4886f7", 1), /steps/);
  });
});

describe("resolveDemographicColors", () => {
  const defaultPalette = generateSequentialPalette(
    DEFAULT_DEMOGRAPHIC_BASE_COLOR,
  );

  it("generates the default palette when unset", () => {
    assert.deepEqual(resolveDemographicColors(undefined), defaultPalette);
    assert.deepEqual(resolveDemographicColors(""), defaultPalette);
  });

  it("generates the default palette from #4886f7", () => {
    assert.equal(DEFAULT_DEMOGRAPHIC_BASE_COLOR, "#4886f7");
    for (const color of defaultPalette) {
      assert.equal(parseOklchStep(color).base, "#4886f7");
    }
  });

  it("uses an explicit array as-is", () => {
    const colors = ["#abcdef", "#123456"];
    assert.equal(resolveDemographicColors(colors), colors);
  });

  it("generates a palette from a single hex color", () => {
    assert.deepEqual(
      resolveDemographicColors("#DA3D2E"),
      generateSequentialPalette("#DA3D2E"),
    );
  });

  it("warns and uses the default palette for an invalid value", (t) => {
    const warn = t.mock.method(console, "warn", () => {});
    assert.deepEqual(resolveDemographicColors("blue"), defaultPalette);
    assert.equal(warn.mock.callCount(), 1);
    assert.match(warn.mock.calls[0].arguments[0], /demographic_colors/);
  });
});
