/**
 * p50k_base Encoding (Legacy)
 * Used by older Codex models (code-davinci-002, code-cushman-001)
 *
 * This encoding has:
 * - 50,257 tokens (smallest vocabulary)
 * - Optimized for code completion
 * - Less efficient for natural language
 *
 * @deprecated This encoding is used by deprecated models.
 * Use o200k_base or cl100k_base for current models.
 */

import { BPETokenizer } from "../bpe.js";
import type { Tokenizer, EncodingName } from "../types.js";

/**
 * Encoding name constant
 */
export const ENCODING_NAME: EncodingName = "p50k_base";

/**
 * Create a p50k_base tokenizer instance
 *
 * @returns Tokenizer instance
 * @deprecated Use createO200kTokenizer or createCL100kTokenizer instead
 */
export function createP50kTokenizer(): Tokenizer {
  const bpe = new BPETokenizer(ENCODING_NAME);

  return {
    encodingName: ENCODING_NAME,
    encode: (text: string) => bpe.encode(text),
    decode: (tokens: number[]) => bpe.decode(tokens),
    countTokens: (text: string) => bpe.countTokens(text),
  };
}

/**
 * Average characters per token for p50k_base
 * Lower than newer encodings due to smaller vocabulary
 */
export const AVERAGE_CHARS_PER_TOKEN = 3.5;

/**
 * Token estimation adjustments for different content types
 * p50k was optimized primarily for code
 */
export const CONTENT_TYPE_MULTIPLIERS: Record<string, number> = {
  prose: 1.15, // Less efficient for natural language
  code: 0.9, // Better for code (its primary use case)
  json: 0.85, // Reasonable for JSON
  markdown: 1.05, // Less efficient for markdown
  html: 0.9, // Reasonable for HTML
  math: 1.25, // Less efficient for math
};

/**
 * Get estimated token count with content-type awareness
 *
 * @param text - Input text
 * @param contentType - Type of content
 * @returns Estimated token count
 * @deprecated Use o200k_base or cl100k_base for current models
 */
export function estimateTokensForContent(
  text: string,
  contentType: keyof typeof CONTENT_TYPE_MULTIPLIERS = "prose",
): number {
  const tokenizer = createP50kTokenizer();
  const baseCount = tokenizer.countTokens(text);
  const multiplier = CONTENT_TYPE_MULTIPLIERS[contentType] ?? 1.15;

  return Math.max(1, Math.round(baseCount * multiplier));
}
