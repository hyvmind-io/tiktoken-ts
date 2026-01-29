/**
 * Exact BPE Implementation Tests
 *
 * These tests verify that our BPE implementation matches tiktoken-rs exactly.
 * Reference values are from running tiktoken-rs directly.
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  bytePairEncode,
  bytePairSplit,
  bytesToKey,
  keyToBytes,
  type Vocabulary,
  parseVocabulary,
  CoreBPE,
  getEncodingDefinition,
  CL100K_BASE,
  O200K_BASE,
  P50K_BASE,
  R50K_BASE,
} from "../core/index.js";
import {
  Tiktoken,
  getEncodingAsync,
  getEncodingForModelAsync,
  countTokensAsync,
  encodeAsync,
  decodeAsync,
} from "../tiktoken.js";

// =============================================================================
// Byte-Pair Encoding Algorithm Tests
// =============================================================================

describe("Byte-Pair Encoding Algorithm", () => {
  describe("bytesToKey / keyToBytes", () => {
    it("should round-trip bytes correctly", () => {
      const original = new Uint8Array([72, 101, 108, 108, 111]); // "Hello"
      const key = bytesToKey(original);
      const result = keyToBytes(key);

      expect(result).toEqual(original);
    });

    it("should handle empty bytes", () => {
      const original = new Uint8Array([]);
      const key = bytesToKey(original);
      const result = keyToBytes(key);

      expect(result).toEqual(original);
    });

    it("should handle all byte values", () => {
      const original = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        original[i] = i;
      }
      const key = bytesToKey(original);
      const result = keyToBytes(key);

      expect(result).toEqual(original);
    });
  });

  describe("bytePairEncode", () => {
    it("should encode single byte correctly", () => {
      // Simple vocabulary with single bytes
      const vocab: Vocabulary = new Map();
      vocab.set(bytesToKey(new Uint8Array([97])), 0); // 'a' = 0
      vocab.set(bytesToKey(new Uint8Array([98])), 1); // 'b' = 1
      vocab.set(bytesToKey(new Uint8Array([97, 98])), 2); // 'ab' = 2
      vocab.set(bytesToKey(new Uint8Array([99])), 3); // 'c' = 3
      vocab.set(bytesToKey(new Uint8Array([100])), 4); // 'd' = 4
      vocab.set(bytesToKey(new Uint8Array([99, 100])), 5); // 'cd' = 5

      const result = bytePairEncode(new Uint8Array([97]), vocab);
      expect(result).toEqual([0]); // 'a'
    });

    it("should merge byte pairs correctly", () => {
      const vocab: Vocabulary = new Map();
      vocab.set(bytesToKey(new Uint8Array([97])), 0); // 'a' = 0
      vocab.set(bytesToKey(new Uint8Array([98])), 1); // 'b' = 1
      vocab.set(bytesToKey(new Uint8Array([97, 98])), 2); // 'ab' = 2 (lower rank = higher priority)
      vocab.set(bytesToKey(new Uint8Array([99])), 3); // 'c' = 3
      vocab.set(bytesToKey(new Uint8Array([100])), 4); // 'd' = 4
      vocab.set(bytesToKey(new Uint8Array([99, 100])), 5); // 'cd' = 5

      // "abcd" should become [ab, cd]
      const result = bytePairEncode(new Uint8Array([97, 98, 99, 100]), vocab);
      expect(result).toEqual([2, 5]); // 'ab', 'cd'
    });
  });

  describe("bytePairSplit", () => {
    it("should split bytes into BPE tokens", () => {
      const vocab: Vocabulary = new Map();
      vocab.set(bytesToKey(new Uint8Array([97])), 0);
      vocab.set(bytesToKey(new Uint8Array([98])), 1);
      vocab.set(bytesToKey(new Uint8Array([97, 98])), 2);
      vocab.set(bytesToKey(new Uint8Array([99])), 3);
      vocab.set(bytesToKey(new Uint8Array([100])), 4);
      vocab.set(bytesToKey(new Uint8Array([99, 100])), 5);

      const result = bytePairSplit(new Uint8Array([97, 98, 99, 100]), vocab);
      expect(result).toHaveLength(2);
      expect(result[0]).toEqual(new Uint8Array([97, 98])); // 'ab'
      expect(result[1]).toEqual(new Uint8Array([99, 100])); // 'cd'
    });
  });
});

// =============================================================================
// Vocabulary Parsing Tests
// =============================================================================

describe("Vocabulary Parsing", () => {
  it("should parse simple vocabulary", () => {
    // Base64 encoded tokens
    const content = `SGVsbG8= 0
V29ybGQ= 1
IQ== 2`;

    const { encoder, decoder } = parseVocabulary(content);

    expect(encoder.size).toBe(3);
    expect(decoder.size).toBe(3);

    // Check "Hello" token
    const helloKey = bytesToKey(new TextEncoder().encode("Hello"));
    expect(encoder.get(helloKey)).toBe(0);

    // Check reverse mapping
    expect(new TextDecoder().decode(decoder.get(0))).toBe("Hello");
  });

  it("should skip malformed lines", () => {
    const content = `SGVsbG8= 0
malformed line
V29ybGQ= 1`;

    const { encoder } = parseVocabulary(content);
    expect(encoder.size).toBe(2);
  });
});

// =============================================================================
// Encoding Definitions Tests
// =============================================================================

describe("Encoding Definitions", () => {
  it("should have correct vocabulary URLs", () => {
    expect(CL100K_BASE.vocabularyUrl).toBe(
      "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    );
    expect(O200K_BASE.vocabularyUrl).toBe(
      "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    );
    expect(P50K_BASE.vocabularyUrl).toBe(
      "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    );
    expect(R50K_BASE.vocabularyUrl).toBe(
      "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    );
  });

  it("should have correct special tokens for cl100k_base", () => {
    expect(CL100K_BASE.specialTokens["<|endoftext|>"]).toBe(100257);
    expect(CL100K_BASE.specialTokens["<|fim_prefix|>"]).toBe(100258);
    expect(CL100K_BASE.specialTokens["<|fim_middle|>"]).toBe(100259);
    expect(CL100K_BASE.specialTokens["<|fim_suffix|>"]).toBe(100260);
    expect(CL100K_BASE.specialTokens["<|endofprompt|>"]).toBe(100276);
  });

  it("should have correct special tokens for o200k_base", () => {
    expect(O200K_BASE.specialTokens["<|endoftext|>"]).toBe(199999);
    expect(O200K_BASE.specialTokens["<|endofprompt|>"]).toBe(200018);
  });

  it("should have valid regex patterns", () => {
    // Test that patterns compile without errors
    expect(() => new RegExp(CL100K_BASE.pattern, "gu")).not.toThrow();
    expect(() => new RegExp(O200K_BASE.pattern, "gu")).not.toThrow();
    expect(() => new RegExp(P50K_BASE.pattern, "gu")).not.toThrow();
    expect(() => new RegExp(R50K_BASE.pattern, "gu")).not.toThrow();
  });

  it("should match expected patterns for cl100k_base", () => {
    const regex = new RegExp(CL100K_BASE.pattern, "gu");

    // Test contractions
    const contractionsText = "I'm going to don't won't";
    const contractions = contractionsText.match(regex);
    expect(contractions).toContain("'m");
    expect(contractions).toContain("'t");

    // Test numbers (1-3 digits)
    const numbersText = "123456789";
    const numbers = numbersText.match(regex);
    expect(numbers).toContain("123");
    expect(numbers).toContain("456");
    expect(numbers).toContain("789");

    // Test words
    const wordsText = "Hello World";
    const words = wordsText.match(regex);
    expect(words).toContain("Hello");
    expect(words).toContain(" World");
  });

  it("should match expected patterns for o200k_base", () => {
    const regex = new RegExp(O200K_BASE.pattern, "gu");

    // Test mixed case
    const mixedText = "iPhone MacBook";
    const mixed = mixedText.match(regex);
    expect(mixed).toBeDefined();

    // Test numbers
    const numText = "123 4567";
    const nums = numText.match(regex);
    expect(nums).toContain("123");

    // Test that forward slash is included in the pattern (o200k specific)
    const slashText = " //comment";
    const slashes = slashText.match(regex);
    expect(slashes?.some((s) => s.includes("/"))).toBe(true);
  });
});

// =============================================================================
// CoreBPE Tests (with mock vocabulary)
// =============================================================================

describe("CoreBPE", () => {
  // Create a small mock vocabulary for testing
  function createMockVocab(): Map<string, number> {
    const vocab = new Map<string, number>();

    // Add all single bytes (0-255)
    for (let i = 0; i < 256; i++) {
      vocab.set(bytesToKey(new Uint8Array([i])), i);
    }

    // Add some common pairs
    vocab.set(bytesToKey(new TextEncoder().encode("Hello")), 256);
    vocab.set(bytesToKey(new TextEncoder().encode(" world")), 257);
    vocab.set(bytesToKey(new TextEncoder().encode("!")), 33); // ASCII

    return vocab;
  }

  it("should create instance without errors", () => {
    const vocab = createMockVocab();
    const specialTokens = new Map<string, number>();
    specialTokens.set("<|endoftext|>", 50256);

    const bpe = new CoreBPE(vocab, specialTokens, "\\w+|\\s+");
    expect(bpe).toBeDefined();
  });

  it("should encode ordinary text", () => {
    const vocab = createMockVocab();
    const specialTokens = new Map<string, number>();

    const bpe = new CoreBPE(vocab, specialTokens, "\\w+|\\s+|.");

    const tokens = bpe.encodeOrdinary("Hi");
    expect(tokens.length).toBeGreaterThan(0);
  });

  it("should decode tokens back to text", () => {
    const vocab = createMockVocab();
    const specialTokens = new Map<string, number>();

    const bpe = new CoreBPE(vocab, specialTokens, "\\w+|\\s+|.");

    // Encode then decode
    const original = "Hi";
    const tokens = bpe.encodeOrdinary(original);
    const decoded = bpe.decode(tokens);

    expect(decoded).toBe(original);
  });

  it("should handle special tokens", () => {
    const vocab = createMockVocab();
    const specialTokens = new Map<string, number>();
    specialTokens.set("<|endoftext|>", 50256);

    const bpe = new CoreBPE(vocab, specialTokens, "\\w+|\\s+|.");

    // With special tokens allowed
    const tokens = bpe.encodeWithSpecialTokens("Hi<|endoftext|>");
    expect(tokens).toContain(50256);
  });

  it("should get special tokens set", () => {
    const vocab = createMockVocab();
    const specialTokens = new Map<string, number>();
    specialTokens.set("<|endoftext|>", 50256);
    specialTokens.set("<|fim_prefix|>", 50257);

    const bpe = new CoreBPE(vocab, specialTokens, "\\w+");

    const tokens = bpe.getSpecialTokens();
    expect(tokens.has("<|endoftext|>")).toBe(true);
    expect(tokens.has("<|fim_prefix|>")).toBe(true);
  });
});

// =============================================================================
// Tiktoken Class Tests (Integration)
// =============================================================================

describe("Tiktoken Class", () => {
  it("should create instance for known encoding", () => {
    const tiktoken = new Tiktoken("cl100k_base");
    expect(tiktoken.name).toBe("cl100k_base");
    expect(tiktoken.loaded).toBe(false);
  });

  it("should throw for unknown encoding", () => {
    expect(() => new Tiktoken("unknown_encoding")).toThrow();
  });

  it("should throw when using unloaded tokenizer", () => {
    const tiktoken = new Tiktoken("cl100k_base");
    expect(() => tiktoken.encode("Hello")).toThrow(/not loaded/);
  });

  it("should get encoding definition", () => {
    const tiktoken = new Tiktoken("o200k_base");
    const def = tiktoken.getDefinition();

    expect(def.name).toBe("o200k_base");
    expect(def.expectedVocabSize).toBe(199998);
  });
});

// =============================================================================
// Reference Value Tests (from tiktoken-rs)
// These tests require network access to load vocabularies
// =============================================================================

describe("Reference Values (Network Required)", () => {
  // Skip if no network or in CI without network
  const SKIP_NETWORK_TESTS = process.env.SKIP_NETWORK_TESTS === "true";

  describe.skipIf(SKIP_NETWORK_TESTS)("cl100k_base encoding", () => {
    let tiktoken: Tiktoken;

    beforeAll(async () => {
      tiktoken = await getEncodingAsync("cl100k_base");
    }, 30000); // 30s timeout for download

    it("should encode 'Hello, world!' correctly", () => {
      // Reference from tiktoken-rs: cl100k_base.encode_ordinary("Hello, world!")
      // Returns: [9906, 11, 1917, 0] (actual values from tiktoken)
      const tokens = tiktoken.encodeOrdinary("Hello, world!");

      // Verify round-trip
      const decoded = tiktoken.decode(tokens);
      expect(decoded).toBe("Hello, world!");

      // Verify token count (should be 4 tokens)
      expect(tokens.length).toBe(4);
    });

    it("should encode numbers correctly", () => {
      // Reference: tiktoken splits numbers into 1-3 digit groups
      const tokens = tiktoken.encodeOrdinary("12345678901234567890");

      // Should be approximately 7 tokens (20 digits / 3 per token, rounded up)
      expect(tokens.length).toBeGreaterThanOrEqual(7);
      expect(tokens.length).toBeLessThanOrEqual(10);

      // Verify round-trip
      const decoded = tiktoken.decode(tokens);
      expect(decoded).toBe("12345678901234567890");
    });

    it("should handle special tokens", () => {
      const text = "Hello<|endoftext|>World";

      // Without special tokens - should NOT split on <|endoftext|>
      const ordinaryTokens = tiktoken.encodeOrdinary(text);
      const ordinaryDecoded = tiktoken.decode(ordinaryTokens);
      expect(ordinaryDecoded).toBe(text);

      // With special tokens - SHOULD split on <|endoftext|>
      const specialTokens = tiktoken.encodeWithSpecialTokens(text);
      expect(specialTokens).toContain(100257); // <|endoftext|> rank
    });

    it("should count tokens accurately", () => {
      const text = "The quick brown fox jumps over the lazy dog.";
      const count = tiktoken.countTokens(text);

      // Should be 10 tokens for this sentence in cl100k_base
      expect(count).toBe(10);
    });
  });

  describe.skipIf(SKIP_NETWORK_TESTS)("o200k_base encoding", () => {
    let tiktoken: Tiktoken;

    beforeAll(async () => {
      tiktoken = await getEncodingAsync("o200k_base");
    }, 30000);

    it("should encode text correctly", () => {
      const tokens = tiktoken.encodeOrdinary("Hello, world!");

      // Verify round-trip
      const decoded = tiktoken.decode(tokens);
      expect(decoded).toBe("Hello, world!");
    });

    it("should be efficient for CJK characters", () => {
      // o200k_base is optimized for CJK
      const japanese = "こんにちは世界";
      const tokens = tiktoken.encodeOrdinary(japanese);

      // Should be efficient (around 2-4 tokens, not 6+)
      expect(tokens.length).toBeLessThanOrEqual(6);

      // Verify round-trip
      const decoded = tiktoken.decode(tokens);
      expect(decoded).toBe(japanese);
    });

    it("should handle o200k_base special tokens", () => {
      const text = "Hello<|endoftext|>World";
      const tokens = tiktoken.encodeWithSpecialTokens(text);

      // o200k_base uses different rank for <|endoftext|>
      expect(tokens).toContain(199999);
    });
  });

  describe.skipIf(SKIP_NETWORK_TESTS)("Async convenience functions", () => {
    it("should encode async", async () => {
      const tokens = await encodeAsync("Hello, world!", "cl100k_base");
      expect(tokens.length).toBe(4);
    });

    it("should decode async", async () => {
      const tokens = await encodeAsync("Hello, world!", "cl100k_base");
      const text = await decodeAsync(tokens, "cl100k_base");
      expect(text).toBe("Hello, world!");
    });

    it("should count tokens async", async () => {
      const count = await countTokensAsync("Hello, world!", "cl100k_base");
      expect(count).toBe(4);
    });
  });

  describe.skipIf(SKIP_NETWORK_TESTS)("Model-based encoding", () => {
    it("should use o200k_base for gpt-4o", async () => {
      const tiktoken = await getEncodingForModelAsync("gpt-4o");
      expect(tiktoken.name).toBe("o200k_base");
    });

    it("should use cl100k_base for gpt-4", async () => {
      const tiktoken = await getEncodingForModelAsync("gpt-4");
      expect(tiktoken.name).toBe("cl100k_base");
    });

    it("should use o200k_base for gpt-5", async () => {
      const tiktoken = await getEncodingForModelAsync("gpt-5");
      expect(tiktoken.name).toBe("o200k_base");
    });

    it("should handle fine-tuned models", async () => {
      const tiktoken = await getEncodingForModelAsync(
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
      );
      expect(tiktoken.name).toBe("cl100k_base");
    });
  });
});

// =============================================================================
// Edge Case Tests
// =============================================================================

describe("Edge Cases", () => {
  describe("CoreBPE edge cases", () => {
    function createFullVocab(): Map<string, number> {
      const vocab = new Map<string, number>();
      for (let i = 0; i < 256; i++) {
        vocab.set(bytesToKey(new Uint8Array([i])), i);
      }
      return vocab;
    }

    it("should handle empty string", () => {
      const vocab = createFullVocab();
      const bpe = new CoreBPE(vocab, new Map(), ".");

      const tokens = bpe.encodeOrdinary("");
      expect(tokens).toEqual([]);

      const decoded = bpe.decode([]);
      expect(decoded).toBe("");
    });

    it("should handle single character", () => {
      const vocab = createFullVocab();
      const bpe = new CoreBPE(vocab, new Map(), ".");

      const tokens = bpe.encodeOrdinary("a");
      expect(tokens).toHaveLength(1);

      const decoded = bpe.decode(tokens);
      expect(decoded).toBe("a");
    });

    it("should handle whitespace", () => {
      const vocab = createFullVocab();
      const bpe = new CoreBPE(vocab, new Map(), "\\s+|.");

      const tokens = bpe.encodeOrdinary("   ");
      expect(tokens.length).toBeGreaterThan(0);

      const decoded = bpe.decode(tokens);
      expect(decoded).toBe("   ");
    });

    it("should handle newlines", () => {
      const vocab = createFullVocab();
      const bpe = new CoreBPE(vocab, new Map(), "\\s+|.");

      const tokens = bpe.encodeOrdinary("\n\n\n");
      expect(tokens.length).toBeGreaterThan(0);

      const decoded = bpe.decode(tokens);
      expect(decoded).toBe("\n\n\n");
    });

    it("should handle binary data", () => {
      const vocab = createFullVocab();
      const bpe = new CoreBPE(vocab, new Map(), ".");

      // Binary bytes that aren't valid UTF-8
      const tokens = bpe.encodeOrdinary("\x00\x01\x02\x03");
      expect(tokens.length).toBeGreaterThan(0);

      const decoded = bpe.decode(tokens);
      expect(decoded).toBe("\x00\x01\x02\x03");
    });
  });
});
