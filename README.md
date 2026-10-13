# tiktoken-ts

A pure TypeScript port of OpenAI's [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs), providing exact BPE (Byte-Pair Encoding) tokenization compatible with OpenAI's models.

## Features

- **Exact BPE tokenization** - Direct port of tiktoken-rs algorithm, produces identical tokens
- **All OpenAI encodings** - `r50k_base`, `p50k_base`, `p50k_edit`, `cl100k_base`, `o200k_base`, `o200k_harmony`
- **Zero dependencies** - Pure TypeScript, works in Node.js and browsers
- **Lazy vocabulary loading** - Vocabularies loaded on-demand from OpenAI CDN (~4-10MB each)
- **Caching** - Vocabularies and tokenizer instances are cached for performance
- **Fast estimation API** - Synchronous heuristic-based counting for quick estimates
- **Model-aware** - Automatic encoding selection for GPT-4, GPT-4o, GPT-5, o-series, and more

## Installation

```bash
npm install tiktoken-ts
```

## Quick Start

### Exact BPE Tokenization (Async)

Use this for exact token counts that match OpenAI's tokenizer:

```typescript
import {
  getEncodingAsync,
  countTokensAsync,
  encodeAsync,
  decodeAsync,
} from "tiktoken-ts";

// Load encoding and tokenize
const tiktoken = await getEncodingAsync("cl100k_base");
const tokens = tiktoken.encode("Hello, world!");
console.log(tokens); // [9906, 11, 1917, 0]

// Decode back to text (round-trip works!)
const text = tiktoken.decode(tokens);
console.log(text); // "Hello, world!"

// Count tokens
const count = tiktoken.countTokens("Hello, world!");
console.log(count); // 4

// Or use convenience functions
const count2 = await countTokensAsync("Hello, world!", "cl100k_base");
const tokens2 = await encodeAsync("Hello!", "o200k_base");
const decoded = await decodeAsync(tokens2, "o200k_base");
```

### For a Specific Model (Async)

```typescript
import {
  getEncodingForModelAsync,
  countTokensForModelAsync,
} from "tiktoken-ts";

// Automatically selects the correct encoding for the model
const tiktoken = await getEncodingForModelAsync("gpt-4o");
const tokens = tiktoken.encode("Hello!");

// Or count directly
const count = await countTokensForModelAsync("Hello!", "gpt-4o");
```

### Token Estimation (Sync)

Use this for fast approximate counts when exact accuracy isn't required:

```typescript
import {
  countTokens,
  estimateMaxTokens,
  getTokenEstimation,
  fitsInContext,
} from "tiktoken-ts";

// Fast token estimation (no vocabulary loading)
const count = countTokens("Hello, world!", { model: "gpt-4o" });

// Estimate safe max_tokens to avoid truncation
const maxTokens = estimateMaxTokens(promptText, "gpt-4o", {
  desiredOutputTokens: 1000,
  safetyMargin: 0.1,
});

// Get detailed estimation with warnings
const estimation = getTokenEstimation(promptText, "gpt-4o");
if (!estimation.fitsInContext) {
  console.warn(estimation.warning);
}

// Check if text fits in context
if (fitsInContext(longText, "gpt-4o", 1000)) {
  // Text fits with 1000 tokens reserved for output
}
```

## API Reference

### Exact BPE API (Async)

#### `getEncodingAsync(encodingName)`

Load an encoding by name. Returns a `Tiktoken` instance.

```typescript
const tiktoken = await getEncodingAsync("cl100k_base");
```

#### `getEncodingForModelAsync(modelName)`

Get the appropriate encoding for a model.

```typescript
const tiktoken = await getEncodingForModelAsync("gpt-4o");
// Uses o200k_base for GPT-4o
```

#### `Tiktoken` Class Methods

```typescript
const tiktoken = await getEncodingAsync("cl100k_base");

// Encode text to tokens
const tokens = tiktoken.encode("Hello!"); // [9906, 0]
const tokens = tiktoken.encodeOrdinary("Hello!"); // Same, no special token handling
const tokens = tiktoken.encodeWithSpecialTokens("<|endoftext|>"); // Handles special tokens

// Decode tokens to text
const text = tiktoken.decode(tokens); // "Hello!"
const bytes = tiktoken.decodeBytes(tokens); // Uint8Array

// Count tokens
const count = tiktoken.countTokens("Hello!"); // 2

// Properties
tiktoken.vocabSize; // Vocabulary size (excluding special tokens)
tiktoken.totalVocabSize; // Total vocabulary size
tiktoken.loaded; // Whether vocabulary is loaded
tiktoken.name; // Encoding name

// Special tokens
tiktoken.getSpecialTokens(); // Set of special token strings
tiktoken.isSpecialToken(100257); // Check if token ID is special
```

#### Convenience Functions

```typescript
// Encode/decode without managing instances
const tokens = await encodeAsync("Hello!", "cl100k_base");
const text = await decodeAsync(tokens, "cl100k_base");
const count = await countTokensAsync("Hello!", "cl100k_base");
const count = await countTokensForModelAsync("Hello!", "gpt-4o");
```

### Estimation API (Sync)

#### `countTokens(text, options?)`

Fast heuristic-based token counting.

```typescript
// With default encoding (o200k_base)
const count = countTokens("Hello, world!");

// With specific model
const count = countTokens("Hello, world!", { model: "gpt-4o" });

// With specific encoding
const count = countTokens("Hello, world!", { encoding: "cl100k_base" });
```

#### `countChatTokens(messages, model?)`

Count tokens in chat messages, including message overhead. Optional `name`, `function_call`, `tool_calls` (`[{ function: { name, arguments } }]`) and `refusal` fields are counted the same way as tiktoken-rs `num_tokens_from_messages`.

```typescript
const messages = [
  { role: "system", content: "You are helpful." },
  { role: "user", content: "Hello!" },
];
const count = countChatTokens(messages, "gpt-4o");
```

#### `estimateMaxTokens(promptText, model, options?)`

Estimate a safe `max_tokens` value for API calls.

```typescript
const maxTokens = estimateMaxTokens(prompt, "gpt-4o", {
  desiredOutputTokens: 1000,
  safetyMargin: 0.1, // 10% safety margin
  minOutputTokens: 100,
  maxOutputTokensCap: 4096,
});
```

#### `getTokenEstimation(promptText, model, options?)`

Get detailed estimation with context fit analysis.

```typescript
const estimation = getTokenEstimation(longPrompt, "gpt-4o", {
  desiredOutputTokens: 2000,
});

console.log({
  promptTokens: estimation.promptTokens,
  recommendedMaxTokens: estimation.recommendedMaxTokens,
  contextLimit: estimation.contextLimit,
  fitsInContext: estimation.fitsInContext,
  warning: estimation.warning,
});
```

### Utility Functions

```typescript
// Check context fit
fitsInContext(text, "gpt-4o", 1000); // reservedOutputTokens

// Truncate to fit
const truncated = truncateToTokenLimit(longText, 1000, "gpt-4o");

// Split into chunks
const chunks = splitIntoChunks(longText, 500, 100, "gpt-4o"); // maxTokens, overlap
```

### Model Configuration

```typescript
import {
  getModelConfig,
  getModelContextLimit,
  getModelMaxOutputTokens,
  getEncodingForModel,
  listModels,
} from "tiktoken-ts";

// Get full model config
const config = getModelConfig("gpt-4o");
// { name: "gpt-4o", encoding: "o200k_base", contextLimit: 128000, maxOutputTokens: 16384, family: "gpt-4o" }

// Get specific values
getModelContextLimit("gpt-4o"); // 128000
getModelMaxOutputTokens("gpt-4o"); // 16384
getEncodingForModel("gpt-4o"); // "o200k_base"

// List all supported models
listModels(); // ["gpt-5.4", "gpt-5.4-pro", ..., "gpt-4o", ...]
```

## Encoding Selection Guide

This section explains **which encoding to use for each model** and **why**.

### Quick Reference Table

| Model Family                     | Encoding            | Type       | Accuracy       | When to Use                       |
| -------------------------------- | ------------------- | ---------- | -------------- | --------------------------------- |
| GPT-4o, GPT-4.1, GPT-5, o-series | `o200k_base`        | Exact BPE  | 100%           | Billing, debugging, decode needed |
| GPT-4, GPT-3.5-turbo             | `cl100k_base`       | Exact BPE  | 100%           | Billing, debugging, decode needed |
| Claude (all versions)            | `claude_estimation` | Estimation | ~80-90% (safe) | Context management, API limits    |
| DeepSeek, Gemini                 | `cl100k_base`       | Estimation | ~70-85%        | Rough estimates only              |
| Legacy GPT-3                     | `r50k_base`         | Exact BPE  | 100%           | Legacy applications               |
| Legacy Codex (code-davinci-\*)   | `p50k_base`         | Exact BPE  | 100%           | Legacy code models                |

### Detailed Encoding Guide

#### `o200k_base` - Modern OpenAI Models (Recommended)

**Use for:** GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-5.x (incl. Codex), o1, o3, o4-mini

```typescript
// Exact tokenization (async, loads vocabulary)
const tiktoken = await getEncodingAsync("o200k_base");
const tokens = tiktoken.encode("Hello!"); // Exact tokens

// Or use model name (auto-selects o200k_base)
const tiktoken = await getEncodingForModelAsync("gpt-4o");
```

**Characteristics:**

- 200,000 token vocabulary
- Most efficient for modern text (~4 chars/token)
- Required for exact billing calculations
- Supports round-trip encode/decode

#### `cl100k_base` - GPT-4 Era Models

**Use for:** GPT-4, GPT-4-turbo, GPT-3.5-turbo, text-embedding-ada-002, text-embedding-3-\*

```typescript
const tiktoken = await getEncodingAsync("cl100k_base");
```

**Characteristics:**

- 100,256 token vocabulary
- Slightly less efficient than o200k_base
- Still widely used for embeddings

#### `claude_estimation` - Anthropic Claude Models

**Use for:** All Claude models (claude-4.5-_, claude-4.1-_, claude-4-_, claude-3.5-_, claude-3-_, claude-2._)

```typescript
// Automatic (recommended)
const count = countTokens("Hello!", { model: "claude-3-5-sonnet" });

// Explicit encoding
const count = countTokens("Hello!", { encoding: "claude_estimation" });

// Content-aware (for code, adds extra safety margin)
import { estimateClaudeTokens } from "tiktoken-ts";
const codeCount = estimateClaudeTokens(pythonCode, "code");
```

**IMPORTANT - Claude is estimation only:**

- Claude uses a **proprietary tokenizer** (not publicly available)
- We apply a **1.25x safety multiplier** to prevent API truncation
- Estimates are intentionally **conservative** (over-count)
- For exact counts, use [Anthropic's Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)

**Why 1.25x multiplier?**

- Research shows Claude produces 16-30% more tokens than GPT-4
- English text: +16%, Math: +21%, Code: +30%
- 1.25x covers worst-case while remaining practical

#### `p50k_base` / `p50k_edit` - Legacy Codex

**Use for:** code-davinci-002, text-davinci-003, text-davinci-edit-001

```typescript
const tiktoken = await getEncodingAsync("p50k_base");
```

#### `r50k_base` - Legacy GPT-3

**Use for:** davinci, curie, babbage, ada (original GPT-3 models)

```typescript
const tiktoken = await getEncodingAsync("r50k_base");
```

### Decision Flowchart

```
Is the model from OpenAI?
├─ YES → Is it GPT-4o, GPT-4.1, GPT-5, or o-series?
│        ├─ YES → Use o200k_base (exact)
│        └─ NO → Is it GPT-4 or GPT-3.5?
│                 ├─ YES → Use cl100k_base (exact)
│                 └─ NO → Is it Codex or text-davinci?
│                          ├─ YES → Use p50k_base (exact)
│                          └─ NO → Use r50k_base (exact)
├─ Is the model from Anthropic (Claude)?
│  └─ YES → Use claude_estimation (safe estimate, 1.25x multiplier)
└─ Other (DeepSeek, Gemini, etc.)
   └─ Use cl100k_base estimation (rough approximation only)
```

### Exact vs Estimation: When to Use Which

| Scenario                  | Use Exact (Async) | Use Estimation (Sync) |
| ------------------------- | ----------------- | --------------------- |
| Billing/cost calculation  | ✅                | ❌                    |
| Debugging tokenization    | ✅                | ❌                    |
| Need to decode tokens     | ✅                | ❌                    |
| Context window management | Either            | ✅ (faster)           |
| Real-time UI feedback     | ❌ (too slow)     | ✅                    |
| Claude models             | N/A               | ✅ (only option)      |
| Batch processing          | ✅                | Either                |

## Supported Encodings

| Encoding            | Vocab Size | Type       | Models                            |
| ------------------- | ---------- | ---------- | --------------------------------- |
| `o200k_base`        | 200,000    | Exact BPE  | GPT-4o, GPT-4.1, GPT-5, o-series  |
| `o200k_harmony`     | 200,000    | Exact BPE  | gpt-oss                           |
| `cl100k_base`       | 100,256    | Exact BPE  | GPT-4, GPT-3.5-turbo, embeddings  |
| `p50k_base`         | 50,257     | Exact BPE  | Code-davinci, text-davinci-003    |
| `p50k_edit`         | 50,257     | Exact BPE  | text-davinci-edit-001             |
| `r50k_base`         | 50,257     | Exact BPE  | GPT-3 (davinci, curie, etc.)      |
| `claude_estimation` | ~22,000\*  | Estimation | All Claude models (safe estimate) |

\*Claude's actual vocabulary size is estimated at ~22,000 based on research, but the encoding uses cl100k_base patterns with a safety multiplier.

## Supported Models

### OpenAI

- **GPT-5 series**: gpt-5.4 (+ pro/mini/nano), gpt-5.2, gpt-5.2-pro, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-turbo
- **Codex**: gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.2-codex, gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5-codex, codex-mini
- **GPT-4.1 series**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano (1M context!)
- **GPT-4o series**: gpt-4o, gpt-4o-mini, chatgpt-4o-latest
- **GPT-4 series**: gpt-4, gpt-4-turbo, gpt-4-32k
- **GPT-3.5 series**: gpt-3.5-turbo, gpt-3.5-turbo-16k
- **o-series**: o1, o1-mini, o3, o3-mini, o3-pro, o4-mini (reasoning models)
- **Embeddings**: text-embedding-ada-002, text-embedding-3-small/large
- **Fine-tuned**: any `ft:<base>:...` name resolves via its base model
- **GPT-6**: context size only (1,050,000); no upstream tokenizer mapping, so estimation uses the `o200k_base` default

### Anthropic Claude (Safe Estimation)

Claude models use a dedicated `claude_estimation` encoding that provides **safe** token estimates with a built-in safety margin. This is designed to prevent API truncation by intentionally over-counting tokens.

**Why is Claude different?**

Claude uses a proprietary tokenizer that is NOT publicly available. Based on research:

- Claude 3+ uses ~22,000 token vocabulary (vs OpenAI's 100K-200K)
- Claude produces 16-30% MORE tokens than GPT-4 for equivalent content
- Average ~3.5 characters per token (vs GPT-4's ~4)

**Our solution:**

The `claude_estimation` encoding applies a **1.25x safety multiplier** to ensure estimates err on over-counting. This prevents API truncation while still providing useful estimates.

```typescript
import {
  countTokens,
  usesClaudeEstimation,
  estimateClaudeTokens,
} from "tiktoken-ts";

// Automatic safe estimation for Claude models
const count = countTokens("Hello, Claude!", { model: "claude-4-5-sonnet" });

// Check if model uses Claude estimation
if (usesClaudeEstimation("claude-3-opus")) {
  console.log("This uses safe Claude estimation");
}

// Content-aware estimation (code has additional +10% multiplier)
const codeCount = estimateClaudeTokens(pythonCode, "code");
```

**For exact Claude token counts**, use Anthropic's official API:

- [Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)

Supported Claude models:

- Claude 4.5, 4.1, 4, 3.5, 3, 2 series

### Others (Estimation only)

- DeepSeek, Gemini (using cl100k_base approximation)

## Accuracy

### Exact BPE API

The async API produces **identical tokens** to OpenAI's tiktoken and tiktoken-rs. Use this when:

- You need exact token counts for billing
- You're debugging tokenization issues
- You need to decode tokens back to text

### Estimation API

The sync estimation API uses heuristics and is:

- **Fast** - No vocabulary loading (instant)
- **Approximate** - Typically within ±10-15% for English (OpenAI models)
- **Conservative** - Tends to slightly over-estimate, safer for API calls

Use estimation when:

- You need quick approximate counts
- You're doing context window management
- Exact counts aren't critical

### Claude Estimation

For Claude models, the estimation is **intentionally conservative** with a 1.25x safety multiplier because:

- Claude's tokenizer is proprietary (not publicly available)
- Claude produces 16-30% more tokens than GPT-4 for equivalent content
- Over-estimation is safer than under-estimation for API limits

For exact Claude counts, use [Anthropic's Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting).

## Browser Usage

The exact BPE API works in browsers but requires fetching vocabulary files (~4-10MB each). Vocabularies are cached after first load.

```typescript
// Works in browsers
const tiktoken = await getEncodingAsync("cl100k_base");
const tokens = tiktoken.encode("Hello!");
```

For bundle-size-sensitive applications, consider:

1. Using the estimation API (zero network requests)
2. Pre-loading vocabularies at app startup
3. Using a service worker to cache vocabularies

## Comparison with Other Libraries

| Library         | Exact BPE | Sync API        | Bundle Size | Dependencies   |
| --------------- | --------- | --------------- | ----------- | -------------- |
| **tiktoken-ts** | ✅        | ✅ (estimation) | ~50KB       | 0              |
| tiktoken (WASM) | ✅        | ✅              | ~4MB        | WASM           |
| gpt-tokenizer   | ✅        | ✅              | ~10MB       | Embedded vocab |
| gpt-3-encoder   | ❌        | ✅              | ~2MB        | r50k only      |

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Type check
npm run typecheck

# Lint
npm run lint

# Format
npm run format
```

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed implementation notes.

Key design decisions:

- Vocabularies loaded from CDN (not embedded) to keep package small
- Dual API: exact async + fast sync estimation
- Direct port of tiktoken-rs BPE algorithm for correctness
- Global caching of vocabularies and instances

## License

MIT

## Credits

- [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) - Original Rust implementation
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's Python implementation
