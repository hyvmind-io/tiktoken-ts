/**
 * Main Tokenizer Implementation
 * Provides high-level tokenization APIs for AI applications
 */

import type {
  Tokenizer,
  EncodingName,
  ChatMessage,
  TokenCountOptions,
  MaxTokensOptions,
  TokenEstimation,
} from "./types.js";
import { BPETokenizer } from "./bpe.js";
import {
  getEncodingForModel,
  getModelContextLimit,
  getModelMaxOutputTokens,
} from "./models.js";

// =============================================================================
// Tokenizer Cache
// =============================================================================

/**
 * Cache of tokenizer instances by encoding name
 */
const tokenizerCache: Map<EncodingName, Tokenizer> = new Map();

/**
 * Get or create a tokenizer for the specified encoding
 *
 * @param encoding - Encoding name
 * @returns Tokenizer instance
 */
function getOrCreateTokenizer(encoding: EncodingName): Tokenizer {
  let tokenizer = tokenizerCache.get(encoding);

  if (!tokenizer) {
    const bpe = new BPETokenizer(encoding);
    tokenizer = {
      encodingName: encoding,
      encode: (text: string) => bpe.encode(text),
      decode: (tokens: number[]) => bpe.decode(tokens),
      countTokens: (text: string) => bpe.countTokens(text),
    };
    tokenizerCache.set(encoding, tokenizer);
  }

  return tokenizer;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Get a tokenizer for a specific encoding
 *
 * @param encodingName - Encoding name (cl100k_base, o200k_base, p50k_base)
 * @returns Tokenizer instance
 *
 * @example
 * ```typescript
 * const tokenizer = getEncoding("o200k_base");
 * const tokens = tokenizer.encode("Hello, world!");
 * console.log(tokens.length); // ~4 tokens
 * ```
 */
export function getEncoding(encodingName: EncodingName): Tokenizer {
  return getOrCreateTokenizer(encodingName);
}

/**
 * Get a tokenizer for a specific model
 *
 * @param model - Model name (e.g., "gpt-4o", "gpt-5-nano")
 * @returns Tokenizer instance configured for the model
 *
 * @example
 * ```typescript
 * const tokenizer = getEncodingForModelName("gpt-4o");
 * const count = tokenizer.countTokens("Hello, world!");
 * console.log(count); // ~4 tokens
 * ```
 */
export function getEncodingForModelName(model: string): Tokenizer {
  const encoding = getEncodingForModel(model);
  return getOrCreateTokenizer(encoding);
}

// =============================================================================
// Token Counting Functions
// =============================================================================

/**
 * Count tokens in text
 *
 * @param text - Input text
 * @param options - Token counting options
 * @returns Token count
 *
 * @example
 * ```typescript
 * // Count with default encoding (o200k_base)
 * const count = countTokens("Hello, world!");
 *
 * // Count with specific model
 * const count = countTokens("Hello, world!", { model: "gpt-4o" });
 *
 * // Count with specific encoding
 * const count = countTokens("Hello, world!", { encoding: "cl100k_base" });
 * ```
 */
export function countTokens(text: string, options?: TokenCountOptions): number {
  const encoding =
    options?.encoding ??
    (options?.model ? getEncodingForModel(options.model) : "o200k_base");
  const tokenizer = getOrCreateTokenizer(encoding);

  return tokenizer.countTokens(text);
}

/**
 * Count tokens in chat messages (includes message overhead)
 *
 * Chat messages have additional tokens for:
 * - Role markers: <|im_start|>role\n ... <|im_end|>
 * - Separator tokens between messages
 *
 * @param messages - Array of chat messages
 * @param model - Model name for accurate counting
 * @returns Total token count including overhead
 *
 * @example
 * ```typescript
 * const messages = [
 *   { role: "system", content: "You are helpful." },
 *   { role: "user", content: "Hello!" }
 * ];
 * const count = countChatTokens(messages, "gpt-4o");
 * ```
 */
export function countChatTokens(
  messages: ChatMessage[],
  model: string = "gpt-4o",
): number {
  const encoding = getEncodingForModel(model);
  const tokenizer = getOrCreateTokenizer(encoding);

  // Overheads from the OpenAI cookbook, as in tiktoken-rs num_tokens_from_messages
  const [tokensPerMessage, tokensPerName] =
    model === "gpt-3.5-turbo-0301" ? [4, -1] : [3, 1];
  const functionCallOverhead = 1;

  let totalTokens = 0;
  for (const message of messages) {
    totalTokens += tokensPerMessage;
    totalTokens += tokenizer.countTokens(message.role);
    totalTokens += tokenizer.countTokens(message.content);
    if (message.name) {
      totalTokens += tokenizer.countTokens(message.name) + tokensPerName;
    }
    const calls = message.tool_calls?.map((call) => call.function) ?? [];
    if (message.function_call) {
      calls.push(message.function_call);
    }
    for (const call of calls) {
      totalTokens +=
        tokenizer.countTokens(call.name) +
        tokenizer.countTokens(call.arguments) +
        functionCallOverhead;
    }
    if (message.refusal) {
      totalTokens += tokenizer.countTokens(message.refusal);
    }
  }

  // Reply priming: <|start|>assistant<|message|>
  totalTokens += 3;

  return totalTokens;
}

/**
 * Count tokens in a prompt with system message
 * Convenience function for common AI playground use case
 *
 * @param systemPrompt - System prompt content
 * @param userPrompt - User prompt content
 * @param model - Model name
 * @returns Total token count
 */
export function countPromptTokens(
  systemPrompt: string,
  userPrompt: string,
  model: string = "gpt-4o",
): number {
  const messages: ChatMessage[] = [];

  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  messages.push({ role: "user", content: userPrompt });

  return countChatTokens(messages, model);
}

// =============================================================================
// Max Tokens Estimation
// =============================================================================

/**
 * Default options for max tokens estimation
 */
const DEFAULT_MAX_TOKENS_OPTIONS: Required<MaxTokensOptions> = {
  desiredOutputTokens: 1000,
  safetyMargin: 0.1,
  minOutputTokens: 100,
  maxOutputTokensCap: 16384,
};

/**
 * Estimate a safe max_tokens value to avoid truncation
 *
 * This function helps prevent the common issue where max_tokens is set too low,
 * causing the API to return an empty response with finish_reason: "length".
 *
 * @param promptText - The full prompt text (or use countChatTokens for messages)
 * @param model - Model name (e.g., "gpt-4o", "gpt-5-nano")
 * @param options - Estimation options
 * @returns Recommended max_tokens value
 *
 * @example
 * ```typescript
 * const prompt = "Write a story about a robot.";
 * const maxTokens = estimateMaxTokens(prompt, "gpt-5-nano", {
 *   desiredOutputTokens: 500,
 *   safetyMargin: 0.15
 * });
 * // Returns a value that ensures the response won't be truncated
 * ```
 */
export function estimateMaxTokens(
  promptText: string,
  model: string,
  options?: MaxTokensOptions,
): number {
  const opts = { ...DEFAULT_MAX_TOKENS_OPTIONS, ...options };

  // Get model configuration
  const contextLimit = getModelContextLimit(model);
  const modelMaxOutput = getModelMaxOutputTokens(model);

  // Count prompt tokens
  const promptTokens = countTokens(promptText, { model });

  // Calculate available tokens (context - prompt - safety buffer)
  const safetyBuffer = Math.ceil(promptTokens * opts.safetyMargin);
  const availableTokens = contextLimit - promptTokens - safetyBuffer;

  // Apply constraints
  let recommendedMaxTokens = Math.min(
    opts.desiredOutputTokens,
    availableTokens,
    modelMaxOutput,
    opts.maxOutputTokensCap,
  );

  // Ensure minimum
  recommendedMaxTokens = Math.max(recommendedMaxTokens, opts.minOutputTokens);

  return Math.round(recommendedMaxTokens);
}

/**
 * Get detailed token estimation with warnings
 *
 * @param promptText - Full prompt text
 * @param model - Model name
 * @param options - Estimation options
 * @returns Detailed token estimation
 *
 * @example
 * ```typescript
 * const estimation = getTokenEstimation(longPrompt, "gpt-4o", {
 *   desiredOutputTokens: 2000
 * });
 *
 * if (!estimation.fitsInContext) {
 *   console.error(estimation.warning);
 * }
 *
 * const response = await openai.chat.completions.create({
 *   model: "gpt-4o",
 *   messages: [...],
 *   max_tokens: estimation.recommendedMaxTokens
 * });
 * ```
 */
export function getTokenEstimation(
  promptText: string,
  model: string,
  options?: MaxTokensOptions,
): TokenEstimation {
  const opts = { ...DEFAULT_MAX_TOKENS_OPTIONS, ...options };

  // Get model configuration
  const contextLimit = getModelContextLimit(model);
  const modelMaxOutput = getModelMaxOutputTokens(model);

  // Count prompt tokens
  const promptTokens = countTokens(promptText, { model });

  // Calculate safety buffer
  const safetyBuffer = Math.ceil(promptTokens * opts.safetyMargin);

  // Calculate available tokens for output
  const availableOutputTokens = Math.max(
    0,
    contextLimit - promptTokens - safetyBuffer,
  );

  // Check if prompt fits in context
  const fitsInContext = promptTokens < contextLimit - opts.minOutputTokens;

  // Calculate recommended max_tokens
  let recommendedMaxTokens = Math.min(
    opts.desiredOutputTokens,
    availableOutputTokens,
    modelMaxOutput,
    opts.maxOutputTokensCap,
  );

  recommendedMaxTokens = Math.max(recommendedMaxTokens, opts.minOutputTokens);

  // Generate warning if needed
  let warning: string | undefined;

  if (!fitsInContext) {
    warning = `Prompt (${promptTokens} tokens) exceeds context limit (${contextLimit}) minus minimum output (${opts.minOutputTokens}). Response may be truncated or fail.`;
  } else if (availableOutputTokens < opts.desiredOutputTokens) {
    warning = `Desired output (${opts.desiredOutputTokens} tokens) reduced to ${recommendedMaxTokens} due to context constraints.`;
  } else if (recommendedMaxTokens === opts.minOutputTokens) {
    warning = `Output limited to minimum (${opts.minOutputTokens} tokens). Consider reducing prompt size.`;
  }

  return {
    promptTokens,
    recommendedMaxTokens: Math.round(recommendedMaxTokens),
    contextLimit,
    fitsInContext,
    availableOutputTokens,
    warning,
  };
}

/**
 * Get token estimation for chat messages
 *
 * @param messages - Array of chat messages
 * @param model - Model name
 * @param options - Estimation options
 * @returns Detailed token estimation
 */
export function getChatTokenEstimation(
  messages: ChatMessage[],
  model: string,
  options?: MaxTokensOptions,
): TokenEstimation {
  const opts = { ...DEFAULT_MAX_TOKENS_OPTIONS, ...options };

  // Get model configuration
  const contextLimit = getModelContextLimit(model);
  const modelMaxOutput = getModelMaxOutputTokens(model);

  // Count prompt tokens (with chat overhead)
  const promptTokens = countChatTokens(messages, model);

  // Calculate safety buffer
  const safetyBuffer = Math.ceil(promptTokens * opts.safetyMargin);

  // Calculate available tokens for output
  const availableOutputTokens = Math.max(
    0,
    contextLimit - promptTokens - safetyBuffer,
  );

  // Check if messages fit in context
  const fitsInContext = promptTokens < contextLimit - opts.minOutputTokens;

  // Calculate recommended max_tokens
  let recommendedMaxTokens = Math.min(
    opts.desiredOutputTokens,
    availableOutputTokens,
    modelMaxOutput,
    opts.maxOutputTokensCap,
  );

  recommendedMaxTokens = Math.max(recommendedMaxTokens, opts.minOutputTokens);

  // Generate warning if needed
  let warning: string | undefined;

  if (!fitsInContext) {
    warning = `Messages (${promptTokens} tokens) exceed context limit (${contextLimit}) minus minimum output (${opts.minOutputTokens}). Response may be truncated or fail.`;
  } else if (availableOutputTokens < opts.desiredOutputTokens) {
    warning = `Desired output (${opts.desiredOutputTokens} tokens) reduced to ${recommendedMaxTokens} due to context constraints.`;
  }

  return {
    promptTokens,
    recommendedMaxTokens: Math.round(recommendedMaxTokens),
    contextLimit,
    fitsInContext,
    availableOutputTokens,
    warning,
  };
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Check if text will fit within model context
 *
 * @param text - Input text
 * @param model - Model name
 * @param reservedOutputTokens - Tokens to reserve for output
 * @returns True if text fits within context
 */
export function fitsInContext(
  text: string,
  model: string,
  reservedOutputTokens: number = 1000,
): boolean {
  const tokens = countTokens(text, { model });
  const contextLimit = getModelContextLimit(model);

  return tokens + reservedOutputTokens <= contextLimit;
}

/**
 * Truncate text to fit within a token limit
 *
 * @param text - Input text
 * @param maxTokens - Maximum tokens allowed
 * @param model - Model name for accurate counting
 * @returns Truncated text
 *
 * @example
 * ```typescript
 * const truncated = truncateToTokenLimit(longText, 1000, "gpt-4o");
 * console.log(countTokens(truncated, { model: "gpt-4o" })); // <= 1000
 * ```
 */
export function truncateToTokenLimit(
  text: string,
  maxTokens: number,
  model: string = "gpt-4o",
): string {
  const encoding = getEncodingForModel(model);
  const tokenizer = getOrCreateTokenizer(encoding);

  const currentTokens = tokenizer.countTokens(text);

  if (currentTokens <= maxTokens) {
    return text;
  }

  // Binary search for the right truncation point
  let low = 0;
  let high = text.length;
  let result = "";

  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    const truncated = text.slice(0, mid);
    const tokens = tokenizer.countTokens(truncated);

    if (tokens <= maxTokens) {
      result = truncated;
      low = mid;
    } else {
      high = mid - 1;
    }
  }

  // Add ellipsis if truncated
  if (result.length < text.length) {
    // Remove last few characters to make room for ellipsis
    while (
      result.length > 0 &&
      tokenizer.countTokens(result + "...") > maxTokens
    ) {
      result = result.slice(0, -1);
    }
    result += "...";
  }

  return result;
}

/**
 * Split text into chunks that fit within token limits
 *
 * @param text - Input text
 * @param maxTokensPerChunk - Maximum tokens per chunk
 * @param overlap - Number of characters to overlap between chunks
 * @param model - Model name
 * @returns Array of text chunks
 */
export function splitIntoChunks(
  text: string,
  maxTokensPerChunk: number,
  overlap: number = 200,
  model: string = "gpt-4o",
): string[] {
  // If text fits in one chunk, return it directly
  const totalTokens = countTokens(text, { model });
  if (totalTokens <= maxTokensPerChunk) {
    return [text.trim()].filter((s) => s.length > 0);
  }

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    // Find the maximum chunk that fits
    let chunkEnd = remaining.length;
    let chunk = remaining.slice(0, chunkEnd);

    while (chunkEnd > 0 && countTokens(chunk, { model }) > maxTokensPerChunk) {
      // Try to break at a sentence or paragraph boundary
      const breakPoints = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "];

      let foundBreak = false;
      for (const bp of breakPoints) {
        const lastBreak = chunk.lastIndexOf(bp, chunkEnd - 1);
        if (lastBreak > 0) {
          chunkEnd = lastBreak + bp.length;
          chunk = remaining.slice(0, chunkEnd);
          if (countTokens(chunk, { model }) <= maxTokensPerChunk) {
            foundBreak = true;
            break;
          }
        }
      }

      if (!foundBreak) {
        // No good break point, just truncate
        chunkEnd = Math.floor(chunkEnd * 0.8);
        chunk = remaining.slice(0, chunkEnd);
      }
    }

    const trimmedChunk = chunk.trim();
    if (trimmedChunk.length > 0) {
      chunks.push(trimmedChunk);
    }

    // Move position forward, accounting for overlap
    // Ensure we always make progress (minimum advance is half of chunk or 1)
    const minAdvance = Math.max(Math.floor(chunkEnd / 2), 1);
    const advance = Math.max(chunkEnd - overlap, minAdvance);
    remaining = remaining.slice(advance);
  }

  return chunks;
}

/**
 * Clear the tokenizer cache
 * Useful for testing or memory management
 */
export function clearTokenizerCache(): void {
  tokenizerCache.clear();
}
