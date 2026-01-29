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

Count tokens in chat messages, including message overhead.

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
listModels(); // ["gpt-5", "gpt-4o", "gpt-4", ...]
```

## Supported Encodings

| Encoding        | Vocab Size | Description         | Models                           |
| --------------- | ---------- | ------------------- | -------------------------------- |
| `o200k_base`    | 200,000    | Latest encoding     | GPT-4o, GPT-4.1, GPT-5, o-series |
| `o200k_harmony` | 200,000    | Open source variant | gpt-oss                          |
| `cl100k_base`   | 100,256    | GPT-4 era           | GPT-4, GPT-3.5-turbo, embeddings |
| `p50k_base`     | 50,257     | Codex era           | Code-davinci, text-davinci-003   |
| `p50k_edit`     | 50,257     | Edit models         | text-davinci-edit-001            |
| `r50k_base`     | 50,257     | Legacy              | GPT-3 (davinci, curie, etc.)     |

## Supported Models

### OpenAI

- **GPT-5 series**: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-turbo
- **GPT-4.1 series**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano (1M context!)
- **GPT-4o series**: gpt-4o, gpt-4o-mini, chatgpt-4o-latest
- **GPT-4 series**: gpt-4, gpt-4-turbo, gpt-4-32k
- **GPT-3.5 series**: gpt-3.5-turbo, gpt-3.5-turbo-16k
- **o-series**: o1, o1-mini, o3, o3-mini, o4-mini (reasoning models)
- **Embeddings**: text-embedding-ada-002, text-embedding-3-small/large
- **Fine-tuned**: ft:gpt-4o, ft:gpt-4, ft:gpt-3.5-turbo

### Anthropic Claude (Estimation only)

Token estimation for Claude models using cl100k_base approximation:

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
- **Approximate** - Typically within ±10-15% for English
- **Conservative** - Tends to slightly over-estimate, safer for API calls

Use estimation when:

- You need quick approximate counts
- You're doing context window management
- Exact counts aren't critical

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
