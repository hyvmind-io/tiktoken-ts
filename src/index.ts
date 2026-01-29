/**
 * tiktoken-ts - A pure TypeScript implementation of OpenAI's tiktoken
 *
 * A self-contained TypeScript tokenizer for AI applications.
 * Implements EXACT BPE (Byte-Pair Encoding) algorithm from tiktoken-rs.
 *
 * @packageDocumentation
 *
 * @example Exact BPE Tokenization (Async)
 * ```typescript
 * import { getEncodingAsync, countTokensAsync } from "tiktoken-ts";
 *
 * // Load encoding and tokenize
 * const tiktoken = await getEncodingAsync("cl100k_base");
 * const tokens = tiktoken.encode("Hello, world!");
 * console.log(tokens); // [15496, 11, 1917, 0]
 *
 * // Decode back to text
 * const text = tiktoken.decode(tokens);
 * console.log(text); // "Hello, world!"
 *
 * // Count tokens
 * const count = await countTokensAsync("Hello, world!", "cl100k_base");
 * console.log(count); // 4
 * ```
 *
 * @example Token Estimation (Sync, uses heuristics)
 * ```typescript
 * import {
 *   countTokens,
 *   estimateMaxTokens,
 *   getTokenEstimation,
 *   getEncodingForModelName,
 *   getModelContextLimit
 * } from "tiktoken-ts";
 *
 * // Count tokens in text (estimation)
 * const tokens = countTokens("Hello, world!", { model: "gpt-4o" });
 *
 * // Estimate safe max_tokens value
 * const maxTokens = estimateMaxTokens(promptText, "gpt-5-nano", {
 *   desiredOutputTokens: 1000,
 *   safetyMargin: 0.1
 * });
 *
 * // Get detailed estimation with warnings
 * const estimation = getTokenEstimation(promptText, "gpt-4o");
 * if (!estimation.fitsInContext) {
 *   console.warn(estimation.warning);
 * }
 * ```
 */

// =============================================================================
// Types
// =============================================================================

export type {
  EncodingName,
  ModelFamily,
  SpecialTokens,
  EncodingConfig,
  Tokenizer,
  ModelConfig,
  TokenEstimation,
  ChatMessage,
  TokenCountOptions,
  MaxTokensOptions,
} from "./types.js";

// =============================================================================
// Core BPE Implementation (Exact tiktoken-rs port)
// =============================================================================

export {
  // Core types
  type Rank,
  type Vocabulary,
  type ReverseVocabulary,

  // BPE algorithm
  bytesToKey,
  keyToBytes,
  bytePairEncode,
  bytePairSplit,

  // Vocabulary loading
  parseVocabulary,
  loadVocabularyFromUrl,
  loadVocabularyFromString,
  clearVocabularyCache,
  isVocabularyCached,
  VOCABULARY_URLS,

  // Core BPE tokenizer
  CoreBPE,
  DecodeKeyError,

  // Encoding definitions
  type EncodingDefinition,
  SPECIAL_TOKENS,
  R50K_BASE,
  P50K_BASE,
  P50K_EDIT,
  CL100K_BASE,
  O200K_BASE,
  O200K_HARMONY,
  ENCODING_DEFINITIONS,
  getEncodingDefinition,
  listEncodingNames,

  // Model mappings (from tiktoken-rs)
  type TokenizerName,
  getTokenizerForModel,
  getContextSize,
  getExactContextSize,
  EXACT_CONTEXT_SIZES,
} from "./core/index.js";

// =============================================================================
// High-Level Tiktoken API (Async, requires vocabulary loading)
// =============================================================================

export {
  // Tiktoken class
  Tiktoken,

  // Factory functions
  getEncoding as getTiktoken,
  getEncodingForModel as getTiktokenForModel,
  getEncodingAsync,
  getEncodingForModelAsync,
  clearTiktokenCache,

  // Async convenience functions
  encodeAsync,
  decodeAsync,
  countTokensAsync,
  countTokensForModelAsync,
} from "./tiktoken.js";

// =============================================================================
// Legacy/Estimation API (Sync, uses heuristics)
// =============================================================================

export {
  // Factory functions
  getEncoding,
  getEncodingForModelName,

  // Token counting (estimation)
  countTokens,
  countChatTokens,
  countPromptTokens,

  // Max tokens estimation
  estimateMaxTokens,
  getTokenEstimation,
  getChatTokenEstimation,

  // Utility functions
  fitsInContext,
  truncateToTokenLimit,
  splitIntoChunks,
  clearTokenizerCache,
} from "./tokenizer.js";

// =============================================================================
// Model Configuration
// =============================================================================

export {
  MODEL_CONFIGS,
  MODEL_ALIASES,
  getModelConfig,
  getEncodingForModel,
  getModelContextLimit,
  getModelMaxOutputTokens,
  getModelFamily,
  usesO200kEncoding,
  listModels,
  listModelsByFamily,
} from "./models.js";

// =============================================================================
// Legacy BPE Implementation (Estimation-based)
// =============================================================================

export {
  BPETokenizer,
  DEFAULT_SPECIAL_TOKENS,
  CL100K_BASE_CONFIG,
  O200K_BASE_CONFIG,
  P50K_BASE_CONFIG,
  getEncodingConfig,
} from "./bpe.js";

// =============================================================================
// Encoding-Specific Exports
// =============================================================================

export {
  // cl100k_base (GPT-4, GPT-3.5-turbo)
  createCL100kTokenizer,
  CL100K_ENCODING_NAME,
  CL100K_AVG_CHARS,
  estimateCL100kTokens,

  // o200k_base (GPT-4o, GPT-4.1, GPT-5)
  createO200kTokenizer,
  O200K_ENCODING_NAME,
  O200K_AVG_CHARS,
  O200K_LANGUAGE_EFFICIENCY,
  estimateO200kTokens,
  estimateO200kTokensForLanguage,

  // p50k_base (Legacy Codex)
  createP50kTokenizer,
  P50K_ENCODING_NAME,
  P50K_AVG_CHARS,
  estimateP50kTokens,
} from "./encodings/index.js";

// =============================================================================
// Utility Functions
// =============================================================================

export {
  stringToBytes,
  bytesToString,
  countCodePoints,
  containsCJK,
  containsEmoji,
  isAscii,
  getTextComplexityMultiplier,
  normalizeWhitespace,
} from "./utils.js";
