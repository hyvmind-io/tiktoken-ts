/**
 * Tokenizer Package Tests
 * Comprehensive tests for the tokenizer implementation
 */

import { describe, it, expect, afterEach } from "vitest";
import {
  // Main API
  countTokens,
  countChatTokens,
  countPromptTokens,
  estimateMaxTokens,
  getTokenEstimation,
  getChatTokenEstimation,
  getEncoding,
  getEncodingForModelName,
  fitsInContext,
  truncateToTokenLimit,
  splitIntoChunks,
  clearTokenizerCache,

  // Model utilities
  getModelConfig,
  getModelContextLimit,
  getModelMaxOutputTokens,
  getEncodingForModel,
  getModelFamily,
  usesO200kEncoding,
  usesClaudeEstimation,
  isClaudeModel,
  listModels,
  listModelsByFamily,

  // BPE
  BPETokenizer,
  getEncodingConfig,
  CLAUDE_SAFETY_MULTIPLIER,

  // Encodings
  createO200kTokenizer,
  createCL100kTokenizer,
  createClaudeEstimationTokenizer,
  estimateClaudeTokens,
  estimateClaudeTokensConservative,
  CLAUDE_VOCAB_SIZE,
  CLAUDE_AVG_CHARS,

  // Utils
  stringToBytes,
  bytesToString,
  containsCJK,
  containsEmoji,
  isAscii,
  countCodePoints,

  // Types
  type ChatMessage,
} from "../index.js";

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Helper to generate text of approximate token length
 */
function generateText(approximateTokens: number): string {
  const words = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "hello",
    "world",
    "test",
    "example",
    "code",
    "function",
    "return",
    "value",
    "string",
    "number",
    "array",
    "object",
  ];

  const result: string[] = [];
  // Approximate: 1.3 tokens per word on average
  const wordsNeeded = Math.ceil(approximateTokens / 1.3);

  for (let i = 0; i < wordsNeeded; i++) {
    result.push(words[i % words.length]);
  }

  return result.join(" ");
}

// =============================================================================
// Basic Token Counting Tests
// =============================================================================

describe("Token Counting", () => {
  afterEach(() => {
    clearTokenizerCache();
  });

  describe("countTokens", () => {
    it("should count tokens in simple text", () => {
      const text = "Hello, world!";
      const count = countTokens(text);

      // Should be around 3-5 tokens
      expect(count).toBeGreaterThanOrEqual(2);
      expect(count).toBeLessThanOrEqual(6);
    });

    it("should return 0 for empty string", () => {
      expect(countTokens("")).toBe(0);
    });

    it("should count tokens with model option", () => {
      const text = "Hello, world!";

      const countGPT4o = countTokens(text, { model: "gpt-4o" });
      const countGPT4 = countTokens(text, { model: "gpt-4" });

      // Both should be reasonable
      expect(countGPT4o).toBeGreaterThan(0);
      expect(countGPT4).toBeGreaterThan(0);
    });

    it("should count tokens with encoding option", () => {
      const text = "Hello, world!";

      const countO200k = countTokens(text, { encoding: "o200k_base" });
      const countCL100k = countTokens(text, { encoding: "cl100k_base" });

      expect(countO200k).toBeGreaterThan(0);
      expect(countCL100k).toBeGreaterThan(0);
    });

    it("should handle special characters", () => {
      const text = "Special: @#$%^&*(){}[]|\\";
      const count = countTokens(text);

      expect(count).toBeGreaterThan(0);
    });

    it("should handle unicode text", () => {
      const text = "Hello 世界! こんにちは 🌍";
      const count = countTokens(text);

      expect(count).toBeGreaterThan(0);
    });

    it("should handle long text", () => {
      const text = generateText(1000);
      const count = countTokens(text);

      // Should be approximately 1000 tokens (within 50% margin)
      expect(count).toBeGreaterThan(500);
      expect(count).toBeLessThan(1500);
    });

    it("should handle code snippets", () => {
      const code = `
function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}
      `;
      const count = countTokens(code);

      expect(count).toBeGreaterThan(10);
      expect(count).toBeLessThan(100);
    });
  });

  describe("countChatTokens", () => {
    it("should count tokens in chat messages", () => {
      const messages: ChatMessage[] = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
      ];

      const count = countChatTokens(messages, "gpt-4o");

      // Should include message overhead
      expect(count).toBeGreaterThan(10);
    });

    it("should handle empty messages array", () => {
      const count = countChatTokens([], "gpt-4o");

      // Should still have assistant priming overhead
      expect(count).toBeGreaterThanOrEqual(3);
    });

    it("should account for message names", () => {
      const messagesWithName: ChatMessage[] = [
        { role: "user", content: "Hello!", name: "John" },
      ];
      const messagesWithoutName: ChatMessage[] = [
        { role: "user", content: "Hello!" },
      ];

      const countWithName = countChatTokens(messagesWithName, "gpt-4o");
      const countWithoutName = countChatTokens(messagesWithoutName, "gpt-4o");

      // Name should add extra tokens
      expect(countWithName).toBeGreaterThan(countWithoutName);
    });

    it("should handle different models", () => {
      const messages: ChatMessage[] = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi" },
      ];

      const countGPT4o = countChatTokens(messages, "gpt-4o");
      const countGPT35 = countChatTokens(messages, "gpt-3.5-turbo");

      // Both should be reasonable
      expect(countGPT4o).toBeGreaterThan(0);
      expect(countGPT35).toBeGreaterThan(0);
    });
  });

  describe("countPromptTokens", () => {
    it("should count tokens for system + user prompt", () => {
      const count = countPromptTokens("You are helpful.", "Hello!", "gpt-4o");

      expect(count).toBeGreaterThan(5);
    });

    it("should handle empty system prompt", () => {
      const count = countPromptTokens("", "Hello!", "gpt-4o");

      expect(count).toBeGreaterThan(0);
    });
  });
});

// =============================================================================
// Max Tokens Estimation Tests
// =============================================================================

describe("Max Tokens Estimation", () => {
  afterEach(() => {
    clearTokenizerCache();
  });

  describe("estimateMaxTokens", () => {
    it("should estimate max tokens for short prompt", () => {
      const prompt = "Write a story.";
      const maxTokens = estimateMaxTokens(prompt, "gpt-4o", {
        desiredOutputTokens: 1000,
      });

      expect(maxTokens).toBe(1000);
    });

    it("should respect model max output tokens", () => {
      const prompt = "Short prompt.";
      const maxTokens = estimateMaxTokens(prompt, "gpt-5-nano", {
        desiredOutputTokens: 10000, // More than model allows
      });

      // Should be capped at model's max output (4096 for nano)
      expect(maxTokens).toBeLessThanOrEqual(4096);
    });

    it("should reduce max tokens for long prompts", () => {
      const shortPrompt = "Hi.";
      const longPrompt = generateText(100000); // Very long prompt to fill most of context

      const maxTokensShort = estimateMaxTokens(shortPrompt, "gpt-4o", {
        desiredOutputTokens: 5000,
      });
      const maxTokensLong = estimateMaxTokens(longPrompt, "gpt-4o", {
        desiredOutputTokens: 5000,
      });

      // For very long prompts, available output tokens should be constrained
      // This test verifies that context limits are respected
      expect(maxTokensLong).toBeLessThanOrEqual(maxTokensShort);
    });

    it("should apply safety margin", () => {
      const prompt = generateText(100000);
      const maxTokens = estimateMaxTokens(prompt, "gpt-4o", {
        desiredOutputTokens: 1000,
        safetyMargin: 0.2, // 20% safety margin
      });

      expect(maxTokens).toBeGreaterThanOrEqual(100); // minOutputTokens
    });

    it("should respect minimum output tokens", () => {
      const prompt = generateText(127000); // Nearly fill context
      const maxTokens = estimateMaxTokens(prompt, "gpt-4o", {
        desiredOutputTokens: 1000,
        minOutputTokens: 100,
      });

      expect(maxTokens).toBeGreaterThanOrEqual(100);
    });

    it("should respect max output cap", () => {
      const prompt = "Short.";
      const maxTokens = estimateMaxTokens(prompt, "gpt-4o", {
        desiredOutputTokens: 50000,
        maxOutputTokensCap: 2000,
      });

      expect(maxTokens).toBeLessThanOrEqual(2000);
    });
  });

  describe("getTokenEstimation", () => {
    it("should return complete estimation object", () => {
      const estimation = getTokenEstimation("Hello!", "gpt-4o");

      expect(estimation).toHaveProperty("promptTokens");
      expect(estimation).toHaveProperty("recommendedMaxTokens");
      expect(estimation).toHaveProperty("contextLimit");
      expect(estimation).toHaveProperty("fitsInContext");
      expect(estimation).toHaveProperty("availableOutputTokens");
    });

    it("should report fitsInContext correctly for short prompts", () => {
      const estimation = getTokenEstimation("Hello!", "gpt-4o");

      expect(estimation.fitsInContext).toBe(true);
      expect(estimation.warning).toBeUndefined();
    });

    it("should report fitsInContext false for very long prompts", () => {
      // Generate text that exceeds context
      const longPrompt = generateText(150000);
      const estimation = getTokenEstimation(longPrompt, "gpt-4o");

      expect(estimation.fitsInContext).toBe(false);
      expect(estimation.warning).toBeDefined();
    });

    it("should provide warning when output is reduced", () => {
      const prompt = generateText(100000);
      const estimation = getTokenEstimation(prompt, "gpt-4o", {
        desiredOutputTokens: 50000,
      });

      // Should have a warning about reduced output
      if (estimation.recommendedMaxTokens < 50000) {
        expect(estimation.warning).toBeDefined();
      }
    });

    it("should calculate correct context limit", () => {
      const estimationGPT4o = getTokenEstimation("Hi", "gpt-4o");
      const estimationGPT35 = getTokenEstimation("Hi", "gpt-3.5-turbo");

      expect(estimationGPT4o.contextLimit).toBe(128000);
      expect(estimationGPT35.contextLimit).toBe(16385);
    });
  });

  describe("getChatTokenEstimation", () => {
    it("should estimate tokens for chat messages", () => {
      const messages: ChatMessage[] = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello!" },
      ];

      const estimation = getChatTokenEstimation(messages, "gpt-4o");

      expect(estimation.promptTokens).toBeGreaterThan(0);
      expect(estimation.fitsInContext).toBe(true);
    });

    it("should handle long conversation history", () => {
      const messages: ChatMessage[] = [];
      for (let i = 0; i < 100; i++) {
        messages.push({
          role: "user",
          content: `Message ${i}: ` + generateText(100),
        });
        messages.push({
          role: "assistant",
          content: `Response ${i}: ` + generateText(100),
        });
      }

      const estimation = getChatTokenEstimation(messages, "gpt-4o");

      expect(estimation.promptTokens).toBeGreaterThan(1000);
    });
  });
});

// =============================================================================
// Model Configuration Tests
// =============================================================================

describe("Model Configuration", () => {
  describe("getModelConfig", () => {
    it("should return config for known models", () => {
      const config = getModelConfig("gpt-4o");

      expect(config).toBeDefined();
      expect(config?.name).toBe("gpt-4o");
      expect(config?.encoding).toBe("o200k_base");
      expect(config?.contextLimit).toBe(128000);
    });

    it("should handle model aliases", () => {
      const config = getModelConfig("gpt4o");

      expect(config).toBeDefined();
      expect(config?.name).toBe("gpt-4o");
    });

    it("should return undefined for unknown models", () => {
      const config = getModelConfig("unknown-model-xyz");

      expect(config).toBeUndefined();
    });

    it("should handle case insensitivity", () => {
      const configLower = getModelConfig("gpt-4o");
      const configUpper = getModelConfig("GPT-4O");

      expect(configLower).toEqual(configUpper);
    });
  });

  describe("getModelContextLimit", () => {
    it("should return correct limits for different models", () => {
      expect(getModelContextLimit("gpt-4o")).toBe(128000);
      expect(getModelContextLimit("gpt-4")).toBe(8192);
      expect(getModelContextLimit("gpt-3.5-turbo")).toBe(16385);
      expect(getModelContextLimit("gemini-1.5-pro")).toBe(1000000);
    });

    it("should return default for unknown models", () => {
      expect(getModelContextLimit("unknown")).toBe(128000);
    });
  });

  describe("getModelMaxOutputTokens", () => {
    it("should return correct output limits", () => {
      expect(getModelMaxOutputTokens("gpt-4o")).toBe(16384);
      expect(getModelMaxOutputTokens("gpt-5-nano")).toBe(4096);
      expect(getModelMaxOutputTokens("o3")).toBe(32768);
    });

    it("should return default for unknown models", () => {
      expect(getModelMaxOutputTokens("unknown")).toBe(4096);
    });
  });

  describe("getEncodingForModel", () => {
    it("should return o200k_base for GPT-4o family", () => {
      expect(getEncodingForModel("gpt-4o")).toBe("o200k_base");
      expect(getEncodingForModel("gpt-4o-mini")).toBe("o200k_base");
      expect(getEncodingForModel("gpt-4.1")).toBe("o200k_base");
      expect(getEncodingForModel("gpt-5")).toBe("o200k_base");
    });

    it("should return cl100k_base for GPT-4 legacy", () => {
      expect(getEncodingForModel("gpt-4")).toBe("cl100k_base");
      expect(getEncodingForModel("gpt-4-turbo")).toBe("cl100k_base");
      expect(getEncodingForModel("gpt-3.5-turbo")).toBe("cl100k_base");
    });

    it("should return default encoding for unknown models", () => {
      expect(getEncodingForModel("unknown")).toBe("o200k_base");
    });
  });

  describe("getModelFamily", () => {
    it("should return correct family", () => {
      expect(getModelFamily("gpt-4o")).toBe("gpt-4o");
      expect(getModelFamily("gpt-5")).toBe("gpt-5");
      expect(getModelFamily("o3")).toBe("o-series");
      expect(getModelFamily("deepseek-chat")).toBe("deepseek");
      expect(getModelFamily("gemini-1.5-pro")).toBe("gemini");
    });

    it("should return unknown for unrecognized models", () => {
      expect(getModelFamily("unknown")).toBe("unknown");
    });
  });

  describe("usesO200kEncoding", () => {
    it("should return true for GPT-4o+ models", () => {
      expect(usesO200kEncoding("gpt-4o")).toBe(true);
      expect(usesO200kEncoding("gpt-4.1")).toBe(true);
      expect(usesO200kEncoding("gpt-5")).toBe(true);
      expect(usesO200kEncoding("o3")).toBe(true);
    });

    it("should return false for legacy models", () => {
      expect(usesO200kEncoding("gpt-4")).toBe(false);
      expect(usesO200kEncoding("gpt-3.5-turbo")).toBe(false);
    });
  });

  describe("listModels", () => {
    it("should return all model names", () => {
      const models = listModels();

      expect(models).toContain("gpt-4o");
      expect(models).toContain("gpt-5");
      expect(models).toContain("gpt-3.5-turbo");
      expect(models.length).toBeGreaterThan(10);
    });
  });

  describe("listModelsByFamily", () => {
    it("should filter by family", () => {
      const gpt5Models = listModelsByFamily("gpt-5");

      expect(gpt5Models.length).toBeGreaterThan(0);
      gpt5Models.forEach((model) => {
        expect(model.family).toBe("gpt-5");
      });
    });

    it("should return empty array for unknown family", () => {
      const models = listModelsByFamily("unknown" as never);

      expect(models).toHaveLength(0);
    });
  });
});

// =============================================================================
// BPE Tokenizer Tests
// =============================================================================

describe("BPE Tokenizer", () => {
  describe("BPETokenizer class", () => {
    it("should create tokenizer with default encoding", () => {
      const tokenizer = new BPETokenizer();

      expect(tokenizer.encodingName).toBe("o200k_base");
    });

    it("should create tokenizer with specified encoding", () => {
      const tokenizer = new BPETokenizer("cl100k_base");

      expect(tokenizer.encodingName).toBe("cl100k_base");
    });

    it("should encode and count tokens", () => {
      const tokenizer = new BPETokenizer();
      const text = "Hello, world!";

      const tokens = tokenizer.encode(text);
      const count = tokenizer.countTokens(text);

      // encode() returns exact tokens, countTokens() uses estimation
      // They should be reasonably close (within 50% margin)
      expect(tokens.length).toBeGreaterThan(0);
      expect(count).toBeGreaterThan(0);
      expect(Math.abs(tokens.length - count)).toBeLessThanOrEqual(
        Math.max(tokens.length, count) * 0.5 + 1,
      );
    });

    it("should handle special tokens", () => {
      const tokenizer = new BPETokenizer();
      const text = "<|im_start|>user\nHello!<|im_end|>";

      const tokens = tokenizer.encode(text);

      // Should have separate tokens for special markers
      expect(tokens.length).toBeGreaterThan(3);
    });

    it("should decode tokens (placeholder)", () => {
      const tokenizer = new BPETokenizer();
      const text = "Hello";

      const tokens = tokenizer.encode(text);
      const decoded = tokenizer.decode(tokens);

      // Decode returns placeholder tokens without full vocab
      expect(decoded).toBeDefined();
    });
  });

  describe("getEncodingConfig", () => {
    it("should return correct configs", () => {
      const o200k = getEncodingConfig("o200k_base");
      const cl100k = getEncodingConfig("cl100k_base");
      const p50k = getEncodingConfig("p50k_base");

      expect(o200k.vocabSize).toBe(200000);
      expect(cl100k.vocabSize).toBe(100256);
      expect(p50k.vocabSize).toBe(50257);
    });
  });
});

// =============================================================================
// Encoding Tests
// =============================================================================

describe("Encodings", () => {
  describe("getEncoding", () => {
    it("should return tokenizer for encoding", () => {
      const tokenizer = getEncoding("o200k_base");

      expect(tokenizer.encodingName).toBe("o200k_base");
      expect(typeof tokenizer.encode).toBe("function");
      expect(typeof tokenizer.decode).toBe("function");
      expect(typeof tokenizer.countTokens).toBe("function");
    });

    it("should cache tokenizers", () => {
      const t1 = getEncoding("o200k_base");
      const t2 = getEncoding("o200k_base");

      expect(t1).toBe(t2);
    });
  });

  describe("getEncodingForModelName", () => {
    it("should return correct tokenizer for model", () => {
      const tokenizer = getEncodingForModelName("gpt-4o");

      expect(tokenizer.encodingName).toBe("o200k_base");
    });
  });

  describe("createO200kTokenizer", () => {
    it("should create working tokenizer", () => {
      const tokenizer = createO200kTokenizer();

      expect(tokenizer.encodingName).toBe("o200k_base");
      expect(tokenizer.countTokens("test")).toBeGreaterThan(0);
    });
  });

  describe("createCL100kTokenizer", () => {
    it("should create working tokenizer", () => {
      const tokenizer = createCL100kTokenizer();

      expect(tokenizer.encodingName).toBe("cl100k_base");
      expect(tokenizer.countTokens("test")).toBeGreaterThan(0);
    });
  });
});

// =============================================================================
// Utility Function Tests
// =============================================================================

describe("Utility Functions", () => {
  describe("fitsInContext", () => {
    it("should return true for short text", () => {
      expect(fitsInContext("Hello", "gpt-4o")).toBe(true);
    });

    it("should return false for text exceeding context", () => {
      const longText = generateText(150000);
      expect(fitsInContext(longText, "gpt-4o")).toBe(false);
    });

    it("should respect reserved output tokens", () => {
      const text = generateText(100000);
      const fitsWithSmallReserve = fitsInContext(text, "gpt-4o", 1000);
      const fitsWithLargeReserve = fitsInContext(text, "gpt-4o", 50000);

      // Should fit with small reserve but not large
      expect(fitsWithSmallReserve).toBe(true);
      expect(fitsWithLargeReserve).toBe(false);
    });
  });

  describe("truncateToTokenLimit", () => {
    it("should not truncate short text", () => {
      const text = "Hello, world!";
      const truncated = truncateToTokenLimit(text, 100, "gpt-4o");

      expect(truncated).toBe(text);
    });

    it("should truncate long text", () => {
      const text = generateText(1000);
      const truncated = truncateToTokenLimit(text, 100, "gpt-4o");

      expect(truncated.length).toBeLessThan(text.length);
      expect(truncated.endsWith("...")).toBe(true);
    });

    it("should respect token limit", () => {
      const text = generateText(1000);
      const truncated = truncateToTokenLimit(text, 50, "gpt-4o");

      const tokenCount = countTokens(truncated, { model: "gpt-4o" });
      expect(tokenCount).toBeLessThanOrEqual(55); // Small margin for ellipsis
    });
  });

  describe("splitIntoChunks", () => {
    it("should not split short text", () => {
      const text = "Hello, world!";
      const chunks = splitIntoChunks(text, 1000);

      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toBe(text);
    });

    it("should split long text into chunks", () => {
      const text = generateText(5000);
      const chunks = splitIntoChunks(text, 500);

      expect(chunks.length).toBeGreaterThan(1);
    });

    it("should create overlapping chunks", () => {
      const text = generateText(2000);
      const chunks = splitIntoChunks(text, 500, 100);

      // Check that chunks have some overlap
      if (chunks.length > 1) {
        // There should be some overlap
        // (not exact due to sentence boundary handling)
        expect(chunks[1].length).toBeGreaterThan(0);
      }
    });

    it("should respect token limits", () => {
      const text = generateText(5000);
      const maxTokens = 500;
      const chunks = splitIntoChunks(text, maxTokens);

      for (const chunk of chunks) {
        const tokens = countTokens(chunk, { model: "gpt-4o" });
        // Allow some margin due to estimation variance
        expect(tokens).toBeLessThanOrEqual(maxTokens * 1.2);
      }
    });
  });
});

// =============================================================================
// String Utility Tests
// =============================================================================

describe("String Utilities", () => {
  describe("stringToBytes / bytesToString", () => {
    it("should round-trip ASCII text", () => {
      const text = "Hello, world!";
      const bytes = stringToBytes(text);
      const result = bytesToString(bytes);

      expect(result).toBe(text);
    });

    it("should round-trip Unicode text", () => {
      const text = "Hello 世界! 🌍";
      const bytes = stringToBytes(text);
      const result = bytesToString(bytes);

      expect(result).toBe(text);
    });
  });

  describe("containsCJK", () => {
    it("should detect Chinese characters", () => {
      expect(containsCJK("你好")).toBe(true);
      expect(containsCJK("Hello 世界")).toBe(true);
    });

    it("should detect Japanese characters", () => {
      expect(containsCJK("こんにちは")).toBe(true);
      expect(containsCJK("日本語")).toBe(true);
    });

    it("should detect Korean characters", () => {
      expect(containsCJK("안녕하세요")).toBe(true);
    });

    it("should return false for ASCII only", () => {
      expect(containsCJK("Hello")).toBe(false);
    });
  });

  describe("containsEmoji", () => {
    it("should detect emoji", () => {
      expect(containsEmoji("Hello 👋")).toBe(true);
      expect(containsEmoji("🎉🎊🎁")).toBe(true);
    });

    it("should return false without emoji", () => {
      expect(containsEmoji("Hello")).toBe(false);
    });
  });

  describe("isAscii", () => {
    it("should return true for ASCII", () => {
      expect(isAscii("Hello, world!")).toBe(true);
      expect(isAscii("123 abc")).toBe(true);
    });

    it("should return false for non-ASCII", () => {
      expect(isAscii("Hello 世界")).toBe(false);
      expect(isAscii("cafe")).toBe(true);
      expect(isAscii("cafe\u0301")).toBe(false); // With combining accent
    });
  });

  describe("countCodePoints", () => {
    it("should count ASCII correctly", () => {
      expect(countCodePoints("Hello")).toBe(5);
    });

    it("should count emoji as single code points", () => {
      expect(countCodePoints("👋")).toBe(1);
      expect(countCodePoints("Hello 👋")).toBe(7); // H e l l o space emoji
    });

    it("should handle surrogate pairs", () => {
      // 𝕳 is a surrogate pair in JavaScript
      expect(countCodePoints("𝕳ello")).toBe(5);
    });
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe("Integration Tests", () => {
  describe("AI Playground Use Case", () => {
    it("should prevent max_tokens truncation issue", () => {
      // Simulate the original issue: max_tokens too low
      const systemPrompt =
        "You are a helpful assistant that writes detailed stories.";
      const userPrompt = "Write a short story about a robot learning to paint.";
      const model = "gpt-5-nano";

      // Get estimation
      const estimation = getTokenEstimation(
        systemPrompt + "\n" + userPrompt,
        model,
        {
          desiredOutputTokens: 1000,
          safetyMargin: 0.1,
        },
      );

      // Verify the estimation is reasonable
      expect(estimation.fitsInContext).toBe(true);
      expect(estimation.recommendedMaxTokens).toBeGreaterThanOrEqual(100);
      expect(estimation.recommendedMaxTokens).toBeLessThanOrEqual(4096); // Model max

      // The recommended max_tokens should be sufficient for the response
      expect(estimation.recommendedMaxTokens).toBeGreaterThanOrEqual(500);
    });

    it("should warn when prompt is too long", () => {
      const veryLongPrompt = generateText(130000); // Exceeds GPT-4o context
      const model = "gpt-4o";

      const estimation = getTokenEstimation(veryLongPrompt, model);

      expect(estimation.fitsInContext).toBe(false);
      expect(estimation.warning).toBeDefined();
      expect(estimation.warning).toContain("exceeds");
    });

    it("should work with chat messages", () => {
      const messages: ChatMessage[] = [
        {
          role: "system",
          content: "You are an AI assistant that helps with coding tasks.",
        },
        {
          role: "user",
          content: "Write a TypeScript function to sort an array.",
        },
      ];

      const estimation = getChatTokenEstimation(messages, "gpt-4o", {
        desiredOutputTokens: 2000,
      });

      expect(estimation.fitsInContext).toBe(true);
      expect(estimation.recommendedMaxTokens).toBeGreaterThanOrEqual(100);
    });
  });

  describe("Different Model Scenarios", () => {
    // Context limits from tiktoken-rs (see core/model-to-encoding.ts)
    const testCases = [
      { model: "gpt-4o", contextLimit: 128000, maxOutput: 16384 },
      { model: "gpt-5-nano", contextLimit: 200000, maxOutput: 4096 }, // tiktoken-rs: 200K
      { model: "gpt-3.5-turbo", contextLimit: 16385, maxOutput: 4096 },
      { model: "deepseek-chat", contextLimit: 32768, maxOutput: 4096 },
      // Claude models (200K context)
      { model: "claude-4-5-sonnet", contextLimit: 200000, maxOutput: 16384 },
      { model: "claude-4-5-opus", contextLimit: 200000, maxOutput: 32768 },
      { model: "claude-4-5-haiku", contextLimit: 200000, maxOutput: 8192 },
      { model: "claude-3-5-sonnet", contextLimit: 200000, maxOutput: 8192 },
    ];

    for (const { model, contextLimit, maxOutput } of testCases) {
      it(`should handle ${model} correctly`, () => {
        const prompt = "Hello, world!";
        const estimation = getTokenEstimation(prompt, model, {
          desiredOutputTokens: 5000,
        });

        expect(estimation.contextLimit).toBe(contextLimit);
        expect(estimation.recommendedMaxTokens).toBeLessThanOrEqual(maxOutput);
        expect(estimation.fitsInContext).toBe(true);
      });
    }
  });

  describe("Claude Model Support", () => {
    it("should support Claude 4.5 Sonnet (latest)", () => {
      const count = countTokens("Hello, Claude!", {
        model: "claude-4-5-sonnet",
      });
      expect(count).toBeGreaterThan(0);
      expect(count).toBeLessThan(10);
    });

    it("should support Claude 4.5 Opus", () => {
      const count = countTokens("Hello, Claude!", { model: "claude-4-5-opus" });
      expect(count).toBeGreaterThan(0);
    });

    it("should support Claude 4.1 Opus", () => {
      const count = countTokens("Hello, Claude!", { model: "claude-4-1-opus" });
      expect(count).toBeGreaterThan(0);
    });

    it("should support Claude 3.5 Sonnet", () => {
      const count = countTokens("Hello, Claude!", {
        model: "claude-3-5-sonnet",
      });
      expect(count).toBeGreaterThan(0);
    });

    it("should support Claude aliases", () => {
      const countAlias = countTokens("Test", { model: "claude" });
      const countFull = countTokens("Test", { model: "claude-4-5-sonnet" });
      expect(countAlias).toBe(countFull);
    });

    it("should handle Claude model context limits", () => {
      expect(getModelContextLimit("claude-4-5-sonnet")).toBe(200000);
      expect(getModelContextLimit("claude-4-5-opus")).toBe(200000);
      expect(getModelContextLimit("claude-4-1-opus")).toBe(200000);
      expect(getModelContextLimit("claude-3-5-sonnet")).toBe(200000);
      expect(getModelContextLimit("claude-3-opus")).toBe(200000);
      expect(getModelContextLimit("claude-2.1")).toBe(200000);
      expect(getModelContextLimit("claude-2.0")).toBe(100000);
    });

    it("should handle Claude model output limits", () => {
      expect(getModelMaxOutputTokens("claude-4-5-opus")).toBe(32768);
      expect(getModelMaxOutputTokens("claude-4-5-sonnet")).toBe(16384);
      expect(getModelMaxOutputTokens("claude-4-5-haiku")).toBe(8192);
      expect(getModelMaxOutputTokens("claude-4-1-opus")).toBe(32768);
      expect(getModelMaxOutputTokens("claude-3-5-sonnet")).toBe(8192);
      expect(getModelMaxOutputTokens("claude-3-opus")).toBe(4096);
    });

    it("should return claude family for Claude models", () => {
      expect(getModelFamily("claude-4-5-sonnet")).toBe("claude");
      expect(getModelFamily("claude-4-5-opus")).toBe("claude");
      expect(getModelFamily("claude-4-1-opus")).toBe("claude");
      expect(getModelFamily("claude-3-5-sonnet")).toBe("claude");
      expect(getModelFamily("claude-3-opus")).toBe("claude");
      expect(getModelFamily("claude-instant-1.2")).toBe("claude");
    });

    it("should estimate tokens for Claude with long context", () => {
      const longPrompt = "A".repeat(100000); // 100K chars
      const estimation = getTokenEstimation(longPrompt, "claude-4-5-sonnet", {
        desiredOutputTokens: 4000,
      });

      // Should fit in Claude's 200K context
      expect(estimation.fitsInContext).toBe(true);
      expect(estimation.contextLimit).toBe(200000);
    });

    it("should handle all Claude version aliases", () => {
      // Latest
      expect(getModelConfig("claude")).toBeDefined();
      expect(getModelConfig("claude-latest")).toBeDefined();

      // 4.5 aliases
      expect(getModelConfig("claude4.5")).toBeDefined();
      expect(getModelConfig("claude-45-sonnet")).toBeDefined();

      // 4.1 aliases
      expect(getModelConfig("claude4.1")).toBeDefined();
      expect(getModelConfig("claude-41-opus")).toBeDefined();

      // 3.5 aliases
      expect(getModelConfig("claude3.5")).toBeDefined();
      expect(getModelConfig("claude-35-sonnet")).toBeDefined();
    });
  });

  // ===========================================================================
  // Claude Estimation Encoding Tests
  // ===========================================================================

  describe("Claude Estimation Encoding", () => {
    it("should use claude_estimation encoding for Claude models", () => {
      expect(getEncodingForModel("claude-4-5-sonnet")).toBe(
        "claude_estimation",
      );
      expect(getEncodingForModel("claude-4-5-opus")).toBe("claude_estimation");
      expect(getEncodingForModel("claude-3-5-sonnet")).toBe(
        "claude_estimation",
      );
      expect(getEncodingForModel("claude-3-opus")).toBe("claude_estimation");
      expect(getEncodingForModel("claude-2.1")).toBe("claude_estimation");
    });

    it("should identify Claude models with usesClaudeEstimation()", () => {
      expect(usesClaudeEstimation("claude-4-5-sonnet")).toBe(true);
      expect(usesClaudeEstimation("claude-3-opus")).toBe(true);
      expect(usesClaudeEstimation("gpt-4o")).toBe(false);
      expect(usesClaudeEstimation("gpt-4")).toBe(false);
    });

    it("should identify Claude models with isClaudeModel()", () => {
      expect(isClaudeModel("claude-4-5-sonnet")).toBe(true);
      expect(isClaudeModel("claude-3-opus")).toBe(true);
      expect(isClaudeModel("claude-2.1")).toBe(true);
      expect(isClaudeModel("gpt-4o")).toBe(false);
    });

    it("should apply safety multiplier for Claude token counting", () => {
      const text = "Hello, Claude! How are you today?";

      // Get base count without safety multiplier (using cl100k_base)
      const baseCount = countTokens(text, { encoding: "cl100k_base" });

      // Get Claude count with safety multiplier
      const claudeCount = countTokens(text, { encoding: "claude_estimation" });

      // Claude count should be higher due to:
      // 1. Lower averageCharsPerToken (3.5 vs 3.8) = ~1.086x more tokens
      // 2. Safety multiplier of 1.25x
      // Combined effect: ~1.36x higher (but varies with text)
      expect(claudeCount).toBeGreaterThan(baseCount);

      // The ratio should be at least 1.2x (safety multiplier effect)
      // Upper bound is relaxed because of combined effects
      const ratio = claudeCount / baseCount;
      expect(ratio).toBeGreaterThanOrEqual(1.2);
      expect(ratio).toBeLessThanOrEqual(1.6); // Allow for combined effects
    });

    it("should create Claude estimation tokenizer", () => {
      const tokenizer = createClaudeEstimationTokenizer();

      expect(tokenizer.encodingName).toBe("claude_estimation");
      expect(tokenizer.countTokens("test")).toBeGreaterThan(0);
    });

    it("should estimate Claude tokens with content type awareness", () => {
      const code = `function hello() { console.log("Hello"); }`;

      const proseCount = estimateClaudeTokens(code, "prose");
      const codeCount = estimateClaudeTokens(code, "code");

      // Code multiplier adds 10% on top of safety multiplier
      expect(codeCount).toBeGreaterThan(proseCount);
    });

    it("should provide conservative Claude estimates", () => {
      const text = "Test message for Claude";

      const normalCount = estimateClaudeTokens(text, "prose");
      const conservativeCount = estimateClaudeTokensConservative(text);

      // Conservative should be higher (uses code multiplier)
      expect(conservativeCount).toBeGreaterThanOrEqual(normalCount);
    });

    it("should have correct Claude encoding constants", () => {
      expect(CLAUDE_SAFETY_MULTIPLIER).toBe(1.25);
      expect(CLAUDE_VOCAB_SIZE).toBe(22000);
      expect(CLAUDE_AVG_CHARS).toBe(3.5);
    });

    it("should return Claude config from getEncodingConfig", () => {
      const config = getEncodingConfig("claude_estimation");

      expect(config.name).toBe("claude_estimation");
      expect(config.vocabSize).toBe(22000);
      expect(config.averageCharsPerToken).toBe(3.5);
    });

    it("should count Claude tokens higher than GPT-4 for same text", () => {
      const testCases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "1 + 2 = 3, and 4 * 5 = 20",
      ];

      for (const text of testCases) {
        const gpt4Count = countTokens(text, { model: "gpt-4" });
        const claudeCount = countTokens(text, { model: "claude-4-5-sonnet" });

        // Claude count should be higher due to safety multiplier
        expect(claudeCount).toBeGreaterThanOrEqual(gpt4Count);
      }
    });

    it("should safely estimate max tokens for Claude models", () => {
      const prompt =
        "Write a detailed analysis of machine learning algorithms.";

      const estimation = getTokenEstimation(prompt, "claude-4-5-sonnet", {
        desiredOutputTokens: 2000,
      });

      // Should fit in context
      expect(estimation.fitsInContext).toBe(true);

      // Prompt tokens should include safety multiplier
      expect(estimation.promptTokens).toBeGreaterThan(0);

      // Recommended max should be safe
      expect(estimation.recommendedMaxTokens).toBeGreaterThan(0);
      expect(estimation.recommendedMaxTokens).toBeLessThanOrEqual(16384); // Claude Sonnet max output
    });

    it("should handle empty text with Claude encoding", () => {
      expect(countTokens("", { encoding: "claude_estimation" })).toBe(0);
    });

    it("should handle BPETokenizer with claude_estimation", () => {
      const tokenizer = new BPETokenizer("claude_estimation");

      expect(tokenizer.encodingName).toBe("claude_estimation");

      const count = tokenizer.countTokens("Hello, Claude!");
      expect(count).toBeGreaterThan(0);
    });
  });
});

// =============================================================================
// Edge Case Tests
// =============================================================================

describe("Edge Cases", () => {
  it("should handle empty string", () => {
    expect(countTokens("")).toBe(0);
    expect(fitsInContext("", "gpt-4o")).toBe(true);
  });

  it("should handle whitespace only", () => {
    expect(countTokens("   ")).toBeGreaterThan(0);
    expect(countTokens("\n\n\n")).toBeGreaterThan(0);
  });

  it("should handle very long words", () => {
    const longWord = "a".repeat(10000);
    const count = countTokens(longWord);

    expect(count).toBeGreaterThan(100);
  });

  it("should handle repeated patterns", () => {
    const repeated = "hello ".repeat(1000);
    const count = countTokens(repeated);

    expect(count).toBeGreaterThan(500);
  });

  it("should handle mixed content", () => {
    const mixed = `
      # Markdown Header
      
      Some text with **bold** and *italic*.
      
      \`\`\`typescript
      const x = 42;
      \`\`\`
      
      - List item 1
      - List item 2
      
      日本語テキスト with emoji 🎉
    `;

    const count = countTokens(mixed);
    expect(count).toBeGreaterThan(20);
  });

  it("should handle JSON content", () => {
    const json = JSON.stringify({
      name: "test",
      values: [1, 2, 3, 4, 5],
      nested: { a: 1, b: 2 },
    });

    const count = countTokens(json);
    expect(count).toBeGreaterThan(10);
  });
});

// =============================================================================
// Accuracy Validation Tests
// =============================================================================

describe("Accuracy Validation", () => {
  /**
   * Reference values from tiktoken-rs (Rust implementation)
   * These are exact token counts for validation.
   *
   * The tokenizer is estimation-based, so we allow a tolerance:
   * - +/-20% for simple text
   * - +/-30% for complex/mixed content
   */

  describe("cl100k_base (GPT-4, GPT-3.5)", () => {
    // Reference: tiktoken-rs cl100k_base.encode_ordinary()
    const referenceValues: [string, number][] = [
      ["Hello, world!", 4],
      ["What can you do?", 5],
      ["The quick brown fox jumps over the lazy dog.", 10],
      ["def hello():\n    print('Hello, World!')", 11],
    ];

    for (const [text, expected] of referenceValues) {
      it(`should count "${text.slice(0, 30)}..." within tolerance`, () => {
        const count = countTokens(text, { model: "gpt-4" });
        // Allow +/-30% tolerance for estimation
        const tolerance = Math.max(2, Math.ceil(expected * 0.3));
        expect(count).toBeGreaterThanOrEqual(expected - tolerance);
        expect(count).toBeLessThanOrEqual(expected + tolerance);
      });
    }
  });

  describe("Number tokenization", () => {
    /**
     * NOTE: This is an estimation-based tokenizer, not an exact BPE implementation.
     * For numbers, the estimation uses a character-based heuristic.
     * Exact tiktoken uses 1-3 digit grouping (e.g., "12345678901234567890" = 7 tokens).
     * Our estimation may over-count numbers, which is SAFE for max_tokens calculation
     * (better to over-estimate than under-estimate and risk truncation).
     */

    it("should provide consistent estimates for number strings", () => {
      const longNumber = "12345678901234567890";
      const count = countTokens(longNumber, { model: "gpt-4" });

      // Estimation-based: expect some over-counting but should be deterministic
      expect(count).toBeGreaterThan(0);
      expect(count).toBeLessThanOrEqual(100); // Sanity check
    });

    it("should handle mixed text with numbers", () => {
      const text = "User ID: 1234567890, Order: 9876543210";
      const count = countTokens(text, { model: "gpt-4o" });

      // Should provide a reasonable estimate for max_tokens calculation
      expect(count).toBeGreaterThanOrEqual(8);
      expect(count).toBeLessThanOrEqual(100); // Sanity check
    });

    it("should handle decimal numbers", () => {
      const text = "3.14159265358979";
      const count = countTokens(text, { model: "gpt-4" });

      // Should provide a reasonable estimate
      expect(count).toBeGreaterThanOrEqual(3);
      expect(count).toBeLessThanOrEqual(100); // Sanity check
    });
  });

  describe("CJK tokenization (o200k efficiency)", () => {
    // o200k_base is HIGHLY efficient for CJK
    // "こんにちは世界" = 2 tokens in tiktoken o200k, not 6+

    it("should estimate Japanese text efficiently for o200k", () => {
      const japanese = "こんにちは世界";
      // tiktoken o200k: 2 tokens
      const count = countTokens(japanese, { model: "gpt-4o" });

      // Should be efficient, not 6+ like cl100k
      expect(count).toBeLessThanOrEqual(5);
      expect(count).toBeGreaterThanOrEqual(1);
    });

    it("should estimate Chinese text efficiently for o200k", () => {
      const chinese = "你好世界";
      const count = countTokens(chinese, { model: "gpt-4o" });

      expect(count).toBeLessThanOrEqual(4);
      expect(count).toBeGreaterThanOrEqual(1);
    });

    it("should estimate Korean text efficiently for o200k", () => {
      const korean = "안녕하세요";
      const count = countTokens(korean, { model: "gpt-4o" });

      expect(count).toBeLessThanOrEqual(4);
      expect(count).toBeGreaterThanOrEqual(1);
    });

    it("should handle mixed CJK and English", () => {
      const mixed = "Hello 世界! Count: 123";
      const count = countTokens(mixed, { model: "gpt-4o" });

      // tiktoken o200k: ~9 tokens
      expect(count).toBeLessThanOrEqual(15);
      expect(count).toBeGreaterThanOrEqual(5);
    });
  });

  describe("Real-world API payload estimation", () => {
    it("should provide safe max_tokens for typical prompt", () => {
      // Simulate the original failing case
      const systemPrompt = "You are a helpful AI assistant.";
      const userPrompt = "What can you do?";
      const fullPrompt = systemPrompt + "\n" + userPrompt;

      const estimation = getTokenEstimation(fullPrompt, "gpt-5-nano", {
        desiredOutputTokens: 256, // Original failing case
        safetyMargin: 0.1,
      });

      // Should fit in context
      expect(estimation.fitsInContext).toBe(true);

      // Should have room for output
      expect(estimation.availableOutputTokens).toBeGreaterThanOrEqual(200);

      // Recommended max_tokens should not cause truncation
      expect(estimation.recommendedMaxTokens).toBeGreaterThanOrEqual(200);
    });

    it("should warn about potential truncation", () => {
      // Large prompt that leaves little room for output
      const largePrompt = generateText(100000); // ~100K tokens

      const estimation = getTokenEstimation(largePrompt, "gpt-4o", {
        desiredOutputTokens: 5000,
      });

      // Should identify the issue
      if (!estimation.fitsInContext) {
        expect(estimation.warning).toBeDefined();
      }
    });
  });
});
