/**
 * cl100k_base Encoding
 * Used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
 *
 * This encoding has:
 * - 100,256 tokens
 * - Better handling of code and programming languages
 * - Improved whitespace handling
 */

import { BPETokenizer } from "../bpe.js";
import type { Tokenizer, EncodingName } from "../types.js";

/**
 * Encoding name constant
 */
export const ENCODING_NAME: EncodingName = "cl100k_base";

/**
 * Create a cl100k_base tokenizer instance
 *
 * @returns Tokenizer instance
 */
export function createCL100kTokenizer(): Tokenizer {
  const bpe = new BPETokenizer(ENCODING_NAME);

  return {
    encodingName: ENCODING_NAME,
    encode: (text: string) => bpe.encode(text),
    decode: (tokens: number[]) => bpe.decode(tokens),
    countTokens: (text: string) => bpe.countTokens(text),
  };
}

/**
 * Pre-computed token counts for common patterns
 * Used to improve estimation accuracy
 */
export const COMMON_TOKEN_PATTERNS: Record<string, number> = {
  // Contractions
  "'s": 1,
  "'t": 1,
  "'re": 1,
  "'ve": 1,
  "'m": 1,
  "'ll": 1,
  "'d": 1,

  // Common words (1 token each in cl100k)
  the: 1,
  and: 1,
  is: 1,
  are: 1,
  was: 1,
  were: 1,
  will: 1,
  would: 1,
  could: 1,
  should: 1,
  have: 1,
  has: 1,
  had: 1,
  been: 1,
  being: 1,
  this: 1,
  that: 1,
  these: 1,
  those: 1,
  with: 1,
  from: 1,
  into: 1,
  for: 1,

  // Programming keywords (1 token each)
  function: 1,
  const: 1,
  let: 1,
  var: 1,
  return: 1,
  if: 1,
  else: 1,
  while: 1,
  class: 1,
  interface: 1,
  type: 1,
  import: 1,
  export: 1,
  async: 1,
  await: 1,
  true: 1,
  false: 1,
  null: 1,
  undefined: 1,

  // Common symbols
  "=>": 1,
  "===": 1,
  "!==": 1,
  "&&": 1,
  "||": 1,
  "++": 1,
  "--": 1,
  "+=": 1,
  "-=": 1,
  "...": 1,

  // Punctuation
  ".": 1,
  ",": 1,
  ";": 1,
  ":": 1,
  "!": 1,
  "?": 1,
  "(": 1,
  ")": 1,
  "[": 1,
  "]": 1,
  "{": 1,
  "}": 1,
  "<": 1,
  ">": 1,
};

/**
 * Average characters per token for cl100k_base
 * Based on empirical analysis of English text
 */
export const AVERAGE_CHARS_PER_TOKEN = 3.8;

/**
 * Token estimation adjustments for different content types
 */
export const CONTENT_TYPE_MULTIPLIERS: Record<string, number> = {
  prose: 1.0, // Normal English text
  code: 0.85, // Code tends to be more token-efficient
  json: 0.75, // JSON is very structured
  markdown: 0.95, // Markdown has extra syntax tokens
  html: 0.8, // HTML tags are often single tokens
  math: 1.2, // Mathematical notation uses more tokens
};

/**
 * Get estimated token count with content-type awareness
 *
 * @param text - Input text
 * @param contentType - Type of content
 * @returns Estimated token count
 */
export function estimateTokensForContent(
  text: string,
  contentType: keyof typeof CONTENT_TYPE_MULTIPLIERS = "prose",
): number {
  const tokenizer = createCL100kTokenizer();
  const baseCount = tokenizer.countTokens(text);
  const multiplier = CONTENT_TYPE_MULTIPLIERS[contentType] ?? 1.0;

  return Math.max(1, Math.round(baseCount * multiplier));
}
