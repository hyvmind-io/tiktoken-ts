/**
 * o200k_base Encoding
 * Used by GPT-4o, GPT-4.1, GPT-5 models
 *
 * This encoding has:
 * - 200,000 tokens (larger vocabulary)
 * - Better handling of non-English languages
 * - Improved efficiency for common patterns
 * - Better support for multimodal inputs
 */

import { BPETokenizer } from "../bpe.js";
import type { Tokenizer, EncodingName } from "../types.js";

/**
 * Encoding name constant
 */
export const ENCODING_NAME: EncodingName = "o200k_base";

/**
 * Create an o200k_base tokenizer instance
 *
 * @returns Tokenizer instance
 */
export function createO200kTokenizer(): Tokenizer {
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
 * o200k_base has a larger vocabulary, so more words are single tokens
 */
export const COMMON_TOKEN_PATTERNS: Record<string, number> = {
  // Contractions (same as cl100k)
  "'s": 1,
  "'t": 1,
  "'re": 1,
  "'ve": 1,
  "'m": 1,
  "'ll": 1,
  "'d": 1,

  // Common words (more are single tokens in o200k)
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
  because: 1, // Single token in o200k
  however: 1,
  therefore: 1,
  although: 1,
  meanwhile: 1,

  // Programming keywords
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

  // TypeScript/React keywords (more are single tokens)
  React: 1,
  useState: 1,
  useEffect: 1,
  useCallback: 1,
  useMemo: 1,
  useRef: 1,
  useContext: 1,
  Component: 1,
  Fragment: 1,
  children: 1,
  props: 1,

  // Common API terms
  request: 1,
  response: 1,
  error: 1,
  success: 1,
  message: 1,
  data: 1,
  status: 1,
  headers: 1,

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
  "?.": 1, // Optional chaining
  "??": 1, // Nullish coalescing
};

/**
 * Average characters per token for o200k_base
 * Slightly higher than cl100k due to larger vocabulary
 */
export const AVERAGE_CHARS_PER_TOKEN = 4.0;

/**
 * Token estimation adjustments for different content types
 * o200k is generally more efficient across all types
 */
export const CONTENT_TYPE_MULTIPLIERS: Record<string, number> = {
  prose: 0.95, // More efficient for English
  code: 0.8, // Even better for code
  json: 0.7, // Very efficient for JSON
  markdown: 0.9, // Better for markdown
  html: 0.75, // Better for HTML
  math: 1.1, // Still needs more tokens for math
  multilingual: 0.85, // Much better for non-English
};

/**
 * Language-specific efficiency factors for o200k_base
 * The larger vocabulary handles more languages efficiently
 */
export const LANGUAGE_EFFICIENCY: Record<string, number> = {
  en: 1.0, // English (baseline)
  es: 1.05, // Spanish
  fr: 1.05, // French
  de: 1.08, // German (compound words)
  pt: 1.05, // Portuguese
  it: 1.05, // Italian
  nl: 1.08, // Dutch
  ru: 1.2, // Russian (Cyrillic)
  ja: 1.5, // Japanese (needs more tokens)
  zh: 1.4, // Chinese (CJK characters)
  ko: 1.4, // Korean (Hangul)
  ar: 1.3, // Arabic
  hi: 1.3, // Hindi
  vi: 1.15, // Vietnamese
  th: 1.3, // Thai
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
  const tokenizer = createO200kTokenizer();
  const baseCount = tokenizer.countTokens(text);
  const multiplier = CONTENT_TYPE_MULTIPLIERS[contentType] ?? 0.95;

  return Math.max(1, Math.round(baseCount * multiplier));
}

/**
 * Get estimated token count with language awareness
 *
 * @param text - Input text
 * @param language - ISO 639-1 language code
 * @returns Estimated token count
 */
export function estimateTokensForLanguage(
  text: string,
  language: string,
): number {
  const tokenizer = createO200kTokenizer();
  const baseCount = tokenizer.countTokens(text);
  const efficiency = LANGUAGE_EFFICIENCY[language.toLowerCase()] ?? 1.0;

  return Math.max(1, Math.round(baseCount * efficiency));
}
