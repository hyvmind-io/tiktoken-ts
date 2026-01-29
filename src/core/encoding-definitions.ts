/**
 * Encoding Definitions
 *
 * EXACT definitions from tiktoken-rs for all supported encodings.
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/tiktoken_ext/openai_public.rs
 */

import type { Rank } from "./byte-pair-encoding.js";

/**
 * Special token constants
 */
export const SPECIAL_TOKENS = {
  ENDOFTEXT: "<|endoftext|>",
  FIM_PREFIX: "<|fim_prefix|>",
  FIM_MIDDLE: "<|fim_middle|>",
  FIM_SUFFIX: "<|fim_suffix|>",
  ENDOFPROMPT: "<|endofprompt|>",
} as const;

/**
 * Encoding definition interface
 */
export interface EncodingDefinition {
  /** Encoding name */
  name: string;

  /**
   * Regex pattern for splitting text into pieces
   * Note: JavaScript regex patterns differ slightly from Rust
   */
  pattern: string;

  /** Special tokens with their ranks */
  specialTokens: Record<string, Rank>;

  /** URL to download the vocabulary file */
  vocabularyUrl: string;

  /** Expected vocabulary size (for validation) */
  expectedVocabSize: number;
}

/**
 * r50k_base encoding (GPT-2)
 *
 * Used by: GPT-3 davinci, curie, babbage, ada models
 * Vocabulary size: 50,257 tokens
 */
export const R50K_BASE: EncodingDefinition = {
  name: "r50k_base",
  // Pattern from tiktoken-rs:
  // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  pattern:
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  specialTokens: {
    [SPECIAL_TOKENS.ENDOFTEXT]: 50256,
  },
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
  expectedVocabSize: 50256,
};

/**
 * p50k_base encoding
 *
 * Used by: Code models, text-davinci-002, text-davinci-003
 * Vocabulary size: 50,281 tokens
 */
export const P50K_BASE: EncodingDefinition = {
  name: "p50k_base",
  // Same pattern as r50k_base
  pattern:
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  specialTokens: {
    [SPECIAL_TOKENS.ENDOFTEXT]: 50256,
  },
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
  expectedVocabSize: 50280,
};

/**
 * p50k_edit encoding
 *
 * Used by: Edit models like text-davinci-edit-001, code-davinci-edit-001
 * Same vocabulary as p50k_base but with FIM special tokens
 */
export const P50K_EDIT: EncodingDefinition = {
  name: "p50k_edit",
  // Same pattern as p50k_base
  pattern:
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  specialTokens: {
    [SPECIAL_TOKENS.ENDOFTEXT]: 50256,
    [SPECIAL_TOKENS.FIM_PREFIX]: 50281,
    [SPECIAL_TOKENS.FIM_MIDDLE]: 50282,
    [SPECIAL_TOKENS.FIM_SUFFIX]: 50283,
  },
  // Uses the same vocabulary file as p50k_base
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
  expectedVocabSize: 50280,
};

/**
 * cl100k_base encoding
 *
 * Used by: ChatGPT models, GPT-4, GPT-3.5-turbo, text-embedding-ada-002
 * Vocabulary size: 100,256 tokens
 */
export const CL100K_BASE: EncodingDefinition = {
  name: "cl100k_base",
  // Pattern from tiktoken-rs:
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  //
  // JavaScript equivalent - (?i:...) is not supported, so we use case-insensitive alternatives
  // Note: JavaScript doesn't support inline case-insensitive groups (?i:)
  // We need to manually expand the contractions pattern
  pattern:
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
  specialTokens: {
    [SPECIAL_TOKENS.ENDOFTEXT]: 100257,
    [SPECIAL_TOKENS.FIM_PREFIX]: 100258,
    [SPECIAL_TOKENS.FIM_MIDDLE]: 100259,
    [SPECIAL_TOKENS.FIM_SUFFIX]: 100260,
    [SPECIAL_TOKENS.ENDOFPROMPT]: 100276,
  },
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
  expectedVocabSize: 100256,
};

/**
 * o200k_base encoding
 *
 * Used by: GPT-4o, GPT-4.1, GPT-5, o-series models
 * Vocabulary size: 200,000 tokens
 *
 * This encoding has a much larger vocabulary and is optimized for:
 * - Better handling of non-English languages
 * - More efficient tokenization of code
 * - Better multimodal support
 */
export const O200K_BASE: EncodingDefinition = {
  name: "o200k_base",
  // Pattern from tiktoken-rs O200K_BASE_PAT_STR:
  // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
  // |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
  // |\p{N}{1,3}
  // | ?[^\s\p{L}\p{N}]+[\r\n/]*   <-- Note: includes forward slash
  // |\s*[\r\n]+
  // |\s+(?!\S)
  // |\s+
  //
  // JavaScript equivalent - expand the case-insensitive contractions
  pattern: [
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
    "\\p{N}{1,3}",
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*", // Note: includes forward slash
    "\\s*[\\r\\n]+",
    "\\s+(?!\\S)",
    "\\s+",
  ].join("|"),
  specialTokens: {
    [SPECIAL_TOKENS.ENDOFTEXT]: 199999,
    [SPECIAL_TOKENS.ENDOFPROMPT]: 200018,
  },
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
  expectedVocabSize: 199998,
};

/**
 * o200k_harmony encoding
 *
 * Used by: gpt-oss models (open source)
 * Same vocabulary as o200k_base but with additional special tokens
 */
export const O200K_HARMONY: EncodingDefinition = {
  name: "o200k_harmony",
  // Same pattern as o200k_base
  pattern: O200K_BASE.pattern,
  specialTokens: {
    "<|startoftext|>": 199998,
    "<|endoftext|>": 199999,
    "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010,
    "<|reserved_200011|>": 200011,
    "<|call|>": 200012,
    // Reserved tokens from 200013 to 201087 are generated dynamically
    ...generateReservedTokens(200013, 201087),
  },
  // Uses the same vocabulary file as o200k_base
  vocabularyUrl:
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
  expectedVocabSize: 199998,
};

/**
 * Generate reserved special tokens
 */
function generateReservedTokens(
  start: number,
  end: number,
): Record<string, Rank> {
  const tokens: Record<string, Rank> = {};
  for (let i = start; i <= end; i++) {
    tokens[`<|reserved_${i}|>`] = i;
  }
  return tokens;
}

/**
 * All encoding definitions
 */
export const ENCODING_DEFINITIONS: Record<string, EncodingDefinition> = {
  r50k_base: R50K_BASE,
  p50k_base: P50K_BASE,
  p50k_edit: P50K_EDIT,
  cl100k_base: CL100K_BASE,
  o200k_base: O200K_BASE,
  o200k_harmony: O200K_HARMONY,
  // Aliases
  gpt2: R50K_BASE,
};

/**
 * Get encoding definition by name
 */
export function getEncodingDefinition(
  name: string,
): EncodingDefinition | undefined {
  return ENCODING_DEFINITIONS[name];
}

/**
 * List all encoding names
 */
export function listEncodingNames(): string[] {
  return Object.keys(ENCODING_DEFINITIONS);
}
