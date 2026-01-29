/**
 * Tokenizer Types
 * Core interfaces for the tiktoken-ts package
 */

/**
 * Encoding names supported by the tokenizer
 * - r50k_base: GPT-2, GPT-3 davinci/curie/babbage/ada
 * - p50k_base: Codex models, text-davinci-002/003
 * - p50k_edit: Edit models (text-davinci-edit-001, code-davinci-edit-001)
 * - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
 * - o200k_base: GPT-4o, GPT-4.1, GPT-5, o-series models
 * - o200k_harmony: gpt-oss models
 * - claude_estimation: Claude models (estimation only, errs on over-counting for safety)
 */
export type EncodingName =
  | "r50k_base"
  | "p50k_base"
  | "p50k_edit"
  | "cl100k_base"
  | "o200k_base"
  | "o200k_harmony"
  | "claude_estimation"
  | "gpt2";

/**
 * Model family classification for encoding selection
 */
export type ModelFamily =
  | "gpt-5"
  | "gpt-4.1"
  | "gpt-4o"
  | "gpt-4"
  | "gpt-3.5"
  | "o-series"
  | "codex"
  | "embedding"
  | "deepseek"
  | "gemini"
  | "claude"
  | "unknown";

/**
 * Special token types used in chat format
 */
export interface SpecialTokens {
  readonly endOfText: string;
  readonly imStart: string;
  readonly imEnd: string;
  readonly imSep: string;
  readonly toolCall?: string;
  readonly toolResult?: string;
}

/**
 * Configuration for an encoding
 */
export interface EncodingConfig {
  readonly name: EncodingName;
  readonly patternSplit: RegExp;
  readonly specialTokens: SpecialTokens;
  readonly vocabSize: number;
  readonly averageCharsPerToken: number;
}

/**
 * Core tokenizer interface
 */
export interface Tokenizer {
  /**
   * Encode text into token IDs
   * @param text - Input text to encode
   * @returns Array of token IDs
   */
  encode(text: string): number[];

  /**
   * Decode token IDs back to text
   * @param tokens - Array of token IDs
   * @returns Decoded text
   */
  decode(tokens: number[]): string;

  /**
   * Count the number of tokens in text
   * @param text - Input text
   * @returns Token count
   */
  countTokens(text: string): number;

  /**
   * Get the encoding name
   */
  readonly encodingName: EncodingName;
}

/**
 * Model context configuration
 */
export interface ModelConfig {
  readonly name: string;
  readonly encoding: EncodingName;
  readonly contextLimit: number;
  readonly maxOutputTokens: number;
  readonly family: ModelFamily;
}

/**
 * Token estimation result
 */
export interface TokenEstimation {
  /** Estimated prompt tokens */
  promptTokens: number;
  /** Recommended max_tokens value */
  recommendedMaxTokens: number;
  /** Model context limit */
  contextLimit: number;
  /** Whether the prompt fits in context */
  fitsInContext: boolean;
  /** Available tokens for output */
  availableOutputTokens: number;
  /** Warning message if any */
  warning?: string;
}

/**
 * Chat message for token counting
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
}

/**
 * Token counting options
 */
export interface TokenCountOptions {
  /** Model name for accurate counting */
  model?: string;
  /** Encoding name override */
  encoding?: EncodingName;
  /** Include special tokens in count */
  includeSpecialTokens?: boolean;
}

/**
 * Max tokens estimation options
 */
export interface MaxTokensOptions {
  /** Desired output token count */
  desiredOutputTokens?: number;
  /** Safety margin percentage (0.0 to 1.0) */
  safetyMargin?: number;
  /** Minimum output tokens to reserve */
  minOutputTokens?: number;
  /** Maximum output tokens cap */
  maxOutputTokensCap?: number;
}
