# tiktoken-ts Architecture

## Overview

tiktoken-ts is a pure TypeScript implementation of OpenAI's tiktoken tokenizer, providing **exact 1:1 compatibility** with tiktoken-rs. The library offers both an exact BPE implementation (requiring vocabulary loading) and a fast estimation mode for synchronous operations.

## Design Decisions

### 1. Vocabulary Storage Strategy: **Lazy Loading from CDN**

We chose **Option B: Lazy-load from CDN on first use** for the following reasons:

- **Package Size**: Embedding vocabularies would add ~15-20MB to the package. Instead, the package is ~50KB.
- **Browser Compatibility**: Lazy loading works seamlessly in both Node.js and browsers via `fetch()`.
- **Caching**: Vocabularies are cached in memory after first load, so subsequent requests are instant.
- **Offline Support**: Users can pre-load vocabularies or embed them if needed using `loadFromString()`.

**Trade-offs:**

- First tokenization requires network access
- Async API required for exact tokenization

### 2. Async vs Sync API

The library provides **both APIs**:

| API Type              | Use Case                              | Accuracy | Network Required |
| --------------------- | ------------------------------------- | -------- | ---------------- |
| **Async (Exact BPE)** | When exact token counts are critical  | 100%     | Yes (first use)  |
| **Sync (Estimation)** | Fast approximations, context checking | 85-95%   | No               |

**Async API (Exact):**

```typescript
import { getEncodingAsync, countTokensAsync } from "tiktoken-ts";

const tiktoken = await getEncodingAsync("cl100k_base");
const tokens = tiktoken.encode("Hello, world!"); // Exact: [9906, 11, 1917, 0]
```

**Sync API (Estimation):**

```typescript
import { countTokens, getTokenEstimation } from "tiktoken-ts";

const count = countTokens("Hello, world!", { model: "gpt-4" }); // Approximate: ~4
```

### 3. Memory Efficiency

For large vocabularies (o200k_base has 200,000 entries), we use:

- **Map<string, number>** for vocabulary (string keys are byte sequences encoded as strings)
- **Map<number, Uint8Array>** for reverse lookup
- **String keys** instead of Uint8Array keys (JavaScript Maps use reference equality for objects)

The `bytesToKey()` function efficiently converts byte arrays to string keys:

```typescript
function bytesToKey(bytes: Uint8Array): string {
  let key = "";
  for (let i = 0; i < bytes.length; i++) {
    key += String.fromCharCode(bytes[i]);
  }
  return key;
}
```

### 4. Caching Strategy

Two levels of caching:

1. **Vocabulary Cache**: Loaded vocabularies are cached globally
2. **Tiktoken Instance Cache**: Tiktoken instances are cached by encoding name

```typescript
// Singleton pattern - same instance returned
const t1 = getEncoding("cl100k_base");
const t2 = getEncoding("cl100k_base");
t1 === t2; // true
```

## File Structure

```
src/
├── core/                      # Exact BPE implementation
│   ├── byte-pair-encoding.ts  # Core BPE algorithm (_byte_pair_merge, etc.)
│   ├── vocab-loader.ts        # Vocabulary file parsing and loading
│   ├── tiktoken.ts            # CoreBPE class (matches tiktoken-rs)
│   ├── encoding-definitions.ts # Exact encoding configs from tiktoken-rs
│   ├── model-to-encoding.ts   # Model name to encoding mappings
│   └── index.ts               # Core module exports
│
├── tiktoken.ts                # High-level Tiktoken class with async API
├── tokenizer.ts               # Sync estimation API (backward compatible)
├── bpe.ts                     # Legacy estimation-based BPE
├── models.ts                  # Model configurations
├── types.ts                   # TypeScript interfaces
├── utils.ts                   # Utility functions
│
├── encodings/                 # Encoding-specific helpers
│   ├── cl100k-base.ts
│   ├── o200k-base.ts
│   ├── p50k-base.ts
│   └── index.ts
│
├── __tests__/
│   ├── tokenizer.test.ts      # Estimation API tests
│   └── bpe-exact.test.ts      # Exact BPE tests
│
└── index.ts                   # Main exports
```

## Encodings Supported

| Encoding        | Vocab Size | Models                                 | Pattern (JavaScript)   |
| --------------- | ---------- | -------------------------------------- | ---------------------- |
| `r50k_base`     | 50,256     | GPT-2, GPT-3 davinci/curie/babbage/ada | `'s\|'t\|'re\|...      |
| `p50k_base`     | 50,280     | Codex, text-davinci-002/003            | Same as r50k           |
| `p50k_edit`     | 50,280     | text-davinci-edit-001                  | Same + FIM tokens      |
| `cl100k_base`   | 100,256    | GPT-4, GPT-3.5-turbo                   | `(?:'[sS]\|...)`       |
| `o200k_base`    | 200,000    | GPT-4o, GPT-4.1, GPT-5, o1/o3          | Complex Unicode        |
| `o200k_harmony` | 200,000    | gpt-oss                                | Same + reserved tokens |

## Algorithm Details

### BPE Merge Algorithm (`_byte_pair_merge`)

Port of tiktoken-rs's `_byte_pair_merge`:

```typescript
function _bytePairMerge(
  ranks: Vocabulary,
  piece: Uint8Array,
): Array<[number, Rank]> {
  // parts[i] = (start_index, rank_of_pair_at_i)
  const parts: Array<[number, Rank]> = [];

  // Initialize with pair ranks
  let minRank = [MAX, MAX];
  for (let i = 0; i < piece.length - 1; i++) {
    const pairKey = bytesToKey(piece.slice(i, i + 2));
    const rank = ranks.get(pairKey) ?? MAX;
    if (rank < minRank[0]) minRank = [rank, i];
    parts.push([i, rank]);
  }
  parts.push([piece.length - 1, MAX]);
  parts.push([piece.length, MAX]);

  // Merge lowest-ranked pairs until no more merges
  while (minRank[0] !== MAX) {
    const i = minRank[1];
    if (i > 0) parts[i - 1][1] = getRank(parts, i - 1);
    parts[i][1] = getRank(parts, i);
    parts.splice(i + 1, 1);

    // Find new minimum
    minRank = [MAX, MAX];
    for (let j = 0; j < parts.length - 1; j++) {
      if (parts[j][1] < minRank[0]) minRank = [parts[j][1], j];
    }
  }

  return parts;
}
```

### Special Token Handling

Special tokens are handled separately from the BPE vocabulary:

1. `encodeOrdinary(text)` - Ignores special tokens (treats them as regular text)
2. `encode(text, allowedSpecial)` - Only processes allowed special tokens
3. `encodeWithSpecialTokens(text)` - Processes all special tokens

## Regex Patterns

### cl100k_base Pattern

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
|[^\r\n\p{L}\p{N}]?\p{L}+
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]+[\r\n]*
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

Note: JavaScript doesn't support `(?i:...)` inline case-insensitive groups, so we expand them manually.

### o200k_base Pattern

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|...)?
|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|...)?
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]+[\r\n/]*   <-- Note: includes forward slash!
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

## Context Sizes (from tiktoken-rs)

Key model context sizes:

| Model         | Context Size |
| ------------- | ------------ |
| gpt-5         | 400,000      |
| gpt-5-mini    | 200,000      |
| gpt-4.1       | 1,047,576    |
| gpt-4o        | 128,000      |
| gpt-4-turbo   | 128,000      |
| gpt-4         | 8,192        |
| gpt-4-32k     | 32,768       |
| gpt-3.5-turbo | 16,385       |
| o1/o3/o4      | 200,000      |

## Migration from Estimation to Exact Mode

### Before (Estimation)

```typescript
import { countTokens } from "tiktoken-ts";

const count = countTokens("Hello, world!", { model: "gpt-4" });
// Approximate: ~4 tokens
```

### After (Exact)

```typescript
import { getEncodingAsync } from "tiktoken-ts";

const tiktoken = await getEncodingAsync("cl100k_base");
const count = tiktoken.countTokens("Hello, world!");
// Exact: 4 tokens
```

### Hybrid Approach

```typescript
import { countTokens, getEncodingAsync } from "tiktoken-ts";

// Use estimation for quick checks
if (countTokens(text, { model }) > limit * 0.9) {
  // Load exact encoder for precise check
  const tiktoken = await getEncodingAsync("cl100k_base");
  return tiktoken.countTokens(text);
}
```

## API Reference

### Exact BPE API (Async)

```typescript
// Get encoder
const tiktoken = await getEncodingAsync("cl100k_base");
const tiktoken = await getEncodingForModelAsync("gpt-4o");

// Encode
tiktoken.encode("Hello"); // With no special tokens
tiktoken.encodeOrdinary("Hello"); // Ignore special tokens
tiktoken.encodeWithSpecialTokens("Hello<|endoftext|>");

// Decode
tiktoken.decode([9906, 11]); // "Hello,"
tiktoken.decodeBytes([9906]); // Uint8Array

// Count
tiktoken.countTokens("Hello"); // 1
```

### Estimation API (Sync)

```typescript
// Count (estimation)
countTokens("Hello", { model: "gpt-4" });
countChatTokens(messages, "gpt-4o");

// Max tokens estimation
estimateMaxTokens(prompt, "gpt-4o", { desiredOutputTokens: 1000 });
getTokenEstimation(prompt, "gpt-4o");

// Utilities
fitsInContext(text, "gpt-4o", reservedTokens);
truncateToTokenLimit(text, maxTokens, "gpt-4o");
```

## Performance Characteristics

| Operation        | Time Complexity | Notes                       |
| ---------------- | --------------- | --------------------------- |
| Vocabulary Load  | O(n)            | n = vocab size, network I/O |
| Encode           | O(m \* n)       | m = merges, n = parts       |
| Decode           | O(n)            | n = token count             |
| Count (exact)    | O(m \* n)       | Same as encode              |
| Count (estimate) | O(n)            | n = text length             |

## Browser Support

The library works in all modern browsers with:

- `fetch()` API for vocabulary loading
- `TextEncoder/TextDecoder` for UTF-8
- `atob/btoa` for Base64

For older browsers, polyfills may be needed.

## Testing

```bash
# Run all tests
npm test

# Run exact BPE tests (requires network)
npm test -- --grep "Reference Values"

# Skip network tests
SKIP_NETWORK_TESTS=true npm test
```
