/**
 * Model to Encoding Mappings
 *
 * EXACT mappings from tiktoken-rs for model name to encoding.
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/tokenizer.rs
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/model.rs
 */

/**
 * Tokenizer/Encoding enum values (matching tiktoken-rs)
 */
export type TokenizerName =
  | "o200k_harmony"
  | "o200k_base"
  | "cl100k_base"
  | "p50k_base"
  | "r50k_base"
  | "p50k_edit"
  | "gpt2";

/**
 * Model prefix to tokenizer mapping
 * Checked in order, so more specific prefixes should come first
 *
 * From tiktoken-rs MODEL_PREFIX_TO_TOKENIZER
 */
const MODEL_PREFIX_TO_TOKENIZER: Array<[string, TokenizerName]> = [
  // Reasoning models
  ["o1-", "o200k_base"],
  ["o3-", "o200k_base"],
  ["o4-mini-", "o200k_base"],
  // Chat models
  ["gpt-5", "o200k_base"],
  ["gpt-4.5-", "o200k_base"],
  ["gpt-4.1-", "o200k_base"],
  ["chatgpt-4o-", "o200k_base"],
  ["gpt-4o-", "o200k_base"],
  ["gpt-4-", "cl100k_base"],
  ["gpt-3.5-turbo-", "cl100k_base"],
  ["gpt-35-turbo-", "cl100k_base"], // Azure deployment name
  ["gpt-oss-", "o200k_harmony"],
];

/**
 * Prefixes not present upstream, checked after MODEL_PREFIX_TO_TOKENIZER
 */
const EXTRA_MODEL_PREFIX_TO_TOKENIZER: Array<[string, TokenizerName]> = [
  ["codex-mini", "o200k_base"],
];

/**
 * Exact model name to tokenizer mapping
 *
 * From tiktoken-rs MODEL_TO_TOKENIZER
 */
const MODEL_TO_TOKENIZER: Map<string, TokenizerName> = new Map([
  // Reasoning models
  ["o1", "o200k_base"],
  ["o3", "o200k_base"],
  ["o4-mini", "o200k_base"],
  // Chat models
  ["gpt-5", "o200k_base"],
  ["gpt-4.1", "o200k_base"],
  ["gpt-4o", "o200k_base"],
  ["gpt-4", "cl100k_base"],
  ["gpt-3.5-turbo", "cl100k_base"],
  ["gpt-3.5", "cl100k_base"],
  ["gpt-35-turbo", "cl100k_base"], // Azure deployment name
  // Base models
  ["davinci-002", "cl100k_base"],
  ["babbage-002", "cl100k_base"],
  // Embeddings
  ["text-embedding-ada-002", "cl100k_base"],
  ["text-embedding-3-small", "cl100k_base"],
  ["text-embedding-3-large", "cl100k_base"],
  // DEPRECATED MODELS - Text
  ["text-davinci-003", "p50k_base"],
  ["text-davinci-002", "p50k_base"],
  ["text-davinci-001", "r50k_base"],
  ["text-curie-001", "r50k_base"],
  ["text-babbage-001", "r50k_base"],
  ["text-ada-001", "r50k_base"],
  ["davinci", "r50k_base"],
  ["curie", "r50k_base"],
  ["babbage", "r50k_base"],
  ["ada", "r50k_base"],
  // DEPRECATED MODELS - Code
  ["code-davinci-002", "p50k_base"],
  ["code-davinci-001", "p50k_base"],
  ["code-cushman-002", "p50k_base"],
  ["code-cushman-001", "p50k_base"],
  ["davinci-codex", "p50k_base"],
  ["cushman-codex", "p50k_base"],
  // DEPRECATED MODELS - Edit
  ["text-davinci-edit-001", "p50k_edit"],
  ["code-davinci-edit-001", "p50k_edit"],
  // DEPRECATED MODELS - Old Embeddings
  ["text-similarity-davinci-001", "r50k_base"],
  ["text-similarity-curie-001", "r50k_base"],
  ["text-similarity-babbage-001", "r50k_base"],
  ["text-similarity-ada-001", "r50k_base"],
  ["text-search-davinci-doc-001", "r50k_base"],
  ["text-search-curie-doc-001", "r50k_base"],
  ["text-search-babbage-doc-001", "r50k_base"],
  ["text-search-ada-doc-001", "r50k_base"],
  ["code-search-babbage-code-001", "r50k_base"],
  ["code-search-ada-code-001", "r50k_base"],
  // Open source
  ["gpt2", "gpt2"],
  ["gpt-2", "gpt2"],
]);

/**
 * Get the tokenizer/encoding for a model name
 *
 * This function matches the logic in tiktoken-rs get_tokenizer()
 *
 * @param modelName - The model name
 * @returns The tokenizer name, or undefined if not found
 */
export function getTokenizerForModel(
  modelName: string,
): TokenizerName | undefined {
  // Fine-tuned models resolve via their base model
  if (modelName.startsWith("ft:")) {
    const rest = modelName.slice(3);
    return getTokenizerForModel(rest.split(":")[0] ?? rest);
  }

  const exactMatch = MODEL_TO_TOKENIZER.get(modelName);
  if (exactMatch) {
    return exactMatch;
  }

  for (const [prefix, tokenizer] of MODEL_PREFIX_TO_TOKENIZER) {
    if (modelName.startsWith(prefix)) {
      return tokenizer;
    }
  }
  for (const [prefix, tokenizer] of EXTRA_MODEL_PREFIX_TO_TOKENIZER) {
    if (modelName.startsWith(prefix)) {
      return tokenizer;
    }
  }

  return undefined;
}

/**
 * Context size mapping from tiktoken-rs model.rs
 *
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/model.rs
 */
export function getContextSize(model: string): number {
  return findContextSize(model) ?? 4096;
}

/**
 * Context size for a model, or undefined when the model is not recognized
 * (mirrors tiktoken-rs get_context_size returning Option)
 */
export function findContextSize(model: string): number | undefined {
  // Handle fine-tuned models
  if (model.startsWith("ft:")) {
    const rest = model.slice(3);
    const base = rest.split(":")[0] ?? rest;
    return findContextSize(base);
  }

  // Check prefixes (most specific first)
  if (model.startsWith("gpt-6")) {
    return 1_050_000;
  }
  if (model.startsWith("gpt-5.4-mini") || model.startsWith("gpt-5.4-nano")) {
    return 400_000;
  }
  if (model.startsWith("gpt-5.4")) {
    return 1_050_000;
  }
  if (model.startsWith("gpt-5.3-codex-spark")) {
    return 128_000;
  }
  if (model.startsWith("gpt-5")) {
    return 400_000;
  }
  if (model.startsWith("codex-mini")) {
    return 200_000;
  }
  if (model.startsWith("gpt-oss")) {
    return 131_072;
  }
  if (model.startsWith("o1-mini") || model.startsWith("o1-preview")) {
    return 128_000;
  }
  if (
    model.startsWith("o1") ||
    model.startsWith("o3") ||
    model.startsWith("o4")
  ) {
    return 200_000;
  }
  if (model.startsWith("gpt-4.1")) {
    return 1_047_576;
  }
  if (model.startsWith("chatgpt-4o") || model.startsWith("gpt-4o")) {
    return 128_000;
  }
  if (model.startsWith("gpt-4.5")) {
    return 128_000;
  }
  if (model.startsWith("gpt-4-turbo-")) {
    return 128_000;
  }
  if (model.startsWith("gpt-4-0125")) {
    return 128_000;
  }
  if (model.startsWith("gpt-4-1106")) {
    return 128_000;
  }
  if (model.startsWith("gpt-4-32k")) {
    return 32_768;
  }
  if (model.startsWith("gpt-4")) {
    return 8192;
  }
  if (model.startsWith("gpt-3.5-turbo-0125")) {
    return 16_385;
  }
  if (model.startsWith("gpt-3.5-turbo-1106")) {
    return 16_385;
  }
  if (model.startsWith("gpt-3.5-turbo-16k")) {
    return 16_385;
  }
  if (
    model.startsWith("gpt-3.5-turbo") ||
    model.startsWith("gpt-3.5") ||
    model.startsWith("gpt-35-turbo")
  ) {
    return 16_385;
  }
  if (
    model.startsWith("text-davinci-002") ||
    model.startsWith("text-davinci-003")
  ) {
    return 4097;
  }
  if (
    model.startsWith("ada") ||
    model.startsWith("babbage") ||
    model.startsWith("curie")
  ) {
    return 2049;
  }
  if (model.startsWith("code-cushman-001")) {
    return 2048;
  }
  if (model.startsWith("code-davinci-002")) {
    return 8001;
  }
  if (model.startsWith("davinci")) {
    return 2049;
  }
  if (
    model.startsWith("text-ada-001") ||
    model.startsWith("text-babbage-001") ||
    model.startsWith("text-curie-001")
  ) {
    return 2049;
  }
  if (model.startsWith("text-embedding-ada-002")) {
    return 8192;
  }

  return undefined;
}

/**
 * Extended context limits including specific model versions
 * from tiktoken-rs model.rs get_context_size match statement
 */
export const EXACT_CONTEXT_SIZES: Record<string, number> = {
  // o-series reasoning models
  o1: 200_000,
  "o1-2024-12-17": 200_000,
  "o1-mini": 128_000,
  "o1-mini-2024-09-12": 128_000,
  "o1-preview": 128_000,
  "o1-preview-2024-09-12": 128_000,
  o3: 200_000,
  "o3-mini": 200_000,
  "o3-mini-2025-01-31": 200_000,
  "o4-mini": 200_000,
  "o4-mini-2025-04-16": 200_000,

  // GPT-5 series
  "gpt-5": 400_000,
  "gpt-5-2025-03-22": 400_000,
  "gpt-5-mini": 200_000,
  "gpt-5-mini-2025-03-22": 200_000,
  "gpt-5-turbo": 400_000,
  "gpt-5-turbo-2025-06-09": 400_000,
  "gpt-5-nano": 200_000,
  "gpt-5-nano-2025-09-14": 200_000,

  // GPT-4.5 preview
  "gpt-4.5-preview": 128_000,
  "gpt-4.5-preview-2025-02-27": 128_000,

  // GPT-4.1 series (1M context!)
  "gpt-4.1": 1_047_576,
  "gpt-4.1-2025-04-14": 1_047_576,
  "gpt-4.1-mini": 1_047_576,
  "gpt-4.1-mini-2025-04-14": 1_047_576,
  "gpt-4.1-nano": 1_047_576,
  "gpt-4.1-nano-2025-04-14": 1_047_576,

  // GPT-4o series
  "chatgpt-4o-latest": 128_000,
  "gpt-4o": 128_000,
  "gpt-4o-2024-05-13": 128_000,
  "gpt-4o-2024-08-06": 128_000,
  "gpt-4o-2024-11-20": 128_000,
  "gpt-4o-audio-preview": 128_000,
  "gpt-4o-audio-preview-2024-10-01": 128_000,
  "gpt-4o-audio-preview-2024-12-17": 128_000,
  "gpt-4o-mini": 128_000,
  "gpt-4o-mini-2024-07-18": 128_000,
  "gpt-4o-mini-audio-preview": 128_000,
  "gpt-4o-mini-audio-preview-2024-12-17": 128_000,
  "gpt-4o-realtime-preview": 128_000,
  "gpt-4o-realtime-preview-2024-10-01": 128_000,
  "gpt-4o-realtime-preview-2024-12-17": 128_000,

  // GPT-4 turbo series
  "gpt-4-turbo": 128_000,
  "gpt-4-turbo-2024-04-09": 128_000,
  "gpt-4-turbo-preview": 128_000,
  "gpt-4-0125-preview": 128_000,
  "gpt-4-1106-preview": 128_000,
  "gpt-4-vision-preview": 128_000,
  "gpt-4-1106-vision-preview": 128_000,

  // GPT-4 base
  "gpt-4": 8_192,
  "gpt-4-0613": 8_192,
  "gpt-4-0314": 8_192,
  "gpt-4-32k": 32_768,
  "gpt-4-32k-0613": 32_768,
  "gpt-4-32k-0314": 32_768,

  // GPT-3.5 series
  "gpt-3.5-turbo": 16_385,
  "gpt-3.5-turbo-0125": 16_385,
  "gpt-3.5-turbo-1106": 16_385,
  "gpt-3.5-turbo-0613": 16_385,
  "gpt-3.5-turbo-16k": 16_385,
  "gpt-3.5-turbo-16k-0613": 16_385,
  "gpt-3.5-turbo-instruct": 4_096,
  "gpt-3.5-turbo-instruct-0914": 4_096,
  "gpt-3.5-turbo-0301": 4_096,

  // gpt-oss
  "gpt-oss": 131_072,
  "gpt-oss-2025-03-01": 131_072,
};

/**
 * Get context size with exact match support
 */
export function getExactContextSize(model: string): number {
  // Check exact match first
  const exactSize = EXACT_CONTEXT_SIZES[model];
  if (exactSize !== undefined) {
    return exactSize;
  }

  // Fall back to prefix matching
  return getContextSize(model);
}
