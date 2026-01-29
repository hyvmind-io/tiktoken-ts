# AGENTS.md - AI Agent Guide for tiktoken-ts

This file provides guidance for AI agents working with the tiktoken-ts codebase.

## Repository Overview

tiktoken-ts is a pure TypeScript implementation of OpenAI's tiktoken tokenizer, providing:

- **Exact BPE tokenization** (async, requires vocabulary loading from CDN)
- **Fast estimation API** (sync, heuristic-based, no network required)
- **Claude-specific estimation** with safety multiplier for Anthropic models
- **Model configuration** for context limits and encoding selection

## Encoding Selection for AI Agents

### CRITICAL: Which Encoding to Use

When tokenizing text for LLM API calls, selecting the correct encoding is essential. Use this guide:

#### OpenAI Models (Exact BPE Available)

| Model                                         | Encoding      | API                                |
| --------------------------------------------- | ------------- | ---------------------------------- |
| gpt-4o, gpt-4o-mini, chatgpt-4o-latest        | `o200k_base`  | Async (exact) or Sync (estimation) |
| gpt-4.1, gpt-4.1-mini, gpt-4.1-nano           | `o200k_base`  | Async (exact) or Sync (estimation) |
| gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-turbo    | `o200k_base`  | Async (exact) or Sync (estimation) |
| o1, o1-mini, o1-preview, o3, o3-mini, o4-mini | `o200k_base`  | Async (exact) or Sync (estimation) |
| gpt-4, gpt-4-turbo, gpt-4-32k                 | `cl100k_base` | Async (exact) or Sync (estimation) |
| gpt-3.5-turbo, gpt-3.5-turbo-16k              | `cl100k_base` | Async (exact) or Sync (estimation) |
| text-embedding-ada-002, text-embedding-3-\*   | `cl100k_base` | Async (exact) or Sync (estimation) |
| code-davinci-\*, text-davinci-002/003         | `p50k_base`   | Async (exact) or Sync (estimation) |
| davinci, curie, babbage, ada (GPT-3)          | `r50k_base`   | Async (exact) or Sync (estimation) |

#### Anthropic Claude Models (Estimation Only)

| Model                                   | Encoding            | API       | Notes                   |
| --------------------------------------- | ------------------- | --------- | ----------------------- |
| claude-4.5-_, claude-4.1-_, claude-4-\* | `claude_estimation` | Sync only | 1.25x safety multiplier |
| claude-3.5-_, claude-3-_                | `claude_estimation` | Sync only | 1.25x safety multiplier |
| claude-2._, claude-instant-_            | `claude_estimation` | Sync only | 1.25x safety multiplier |

**Why Claude is different:**

- Claude uses a **proprietary tokenizer** not publicly available
- No vocabulary file exists for exact BPE
- We use `cl100k_base` patterns with a **1.25x safety multiplier**
- Estimates intentionally **over-count** to prevent API truncation
- For exact counts, use [Anthropic's Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)

#### Other Models (Approximation Only)

| Model                                    | Encoding      | Notes                                  |
| ---------------------------------------- | ------------- | -------------------------------------- |
| DeepSeek (deepseek-chat, deepseek-coder) | `cl100k_base` | Rough approximation (~70-85% accurate) |
| Gemini (gemini-1.5-_, gemini-2.0-_)      | `cl100k_base` | Rough approximation (~70-85% accurate) |

### Code Examples for AI Agents

```typescript
// === OPENAI MODELS ===

// For exact token counts (billing, debugging)
const tiktoken = await getEncodingAsync("o200k_base");
const exactCount = tiktoken.countTokens(text);

// For fast estimation (context management)
const estimatedCount = countTokens(text, { model: "gpt-4o" });

// === CLAUDE MODELS ===

// Automatic safe estimation (RECOMMENDED)
const claudeCount = countTokens(text, { model: "claude-3-5-sonnet" });

// Explicit encoding
const claudeCount2 = countTokens(text, { encoding: "claude_estimation" });

// Content-aware (code gets extra +10% safety)
import { estimateClaudeTokens } from "tiktoken-ts";
const codeCount = estimateClaudeTokens(pythonCode, "code");

// === CHECKING ENCODING TYPE ===

import {
  usesClaudeEstimation,
  isClaudeModel,
  getEncodingForModel,
} from "tiktoken-ts";

if (isClaudeModel("claude-3-opus")) {
  // Use estimation API only
  const count = countTokens(text, { model: "claude-3-opus" });
}

if (usesClaudeEstimation("claude-4-5-sonnet")) {
  console.log("Warning: Token count is an estimate with 1.25x safety margin");
}

const encoding = getEncodingForModel("gpt-4o"); // Returns "o200k_base"
```

### Decision Rules for AI Agents

1. **Need exact token count?** → Use async API with appropriate encoding
2. **OpenAI model?** → Use model's encoding (o200k_base for modern, cl100k_base for GPT-4 era)
3. **Claude model?** → Use `claude_estimation` (sync only, includes safety margin)
4. **Need speed over accuracy?** → Use sync estimation API
5. **Unknown model?** → Default to `o200k_base` estimation

### Safety Multipliers Explained

| Encoding                               | Multiplier | Reason                                            |
| -------------------------------------- | ---------- | ------------------------------------------------- |
| OpenAI (o200k_base, cl100k_base, etc.) | 1.0x       | Exact BPE available                               |
| `claude_estimation`                    | 1.25x      | Claude uses 16-30% more tokens than GPT-4         |
| `claude_estimation` + code content     | 1.375x     | Code is particularly inefficient on Claude (+30%) |

## Build & Test Commands

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Type check
npm run typecheck

# Lint
npm run lint

# Format code
npm run format
```

## Project Structure

```
src/
├── core/                        # Exact BPE implementation (tiktoken-rs port)
│   ├── byte-pair-encoding.ts    # Core BPE algorithm
│   ├── vocab-loader.ts          # Vocabulary file parsing
│   ├── tiktoken.ts              # CoreBPE class
│   ├── encoding-definitions.ts  # Exact encoding configs
│   └── model-to-encoding.ts     # Model-to-encoding mappings
│
├── encodings/                   # Encoding-specific implementations
│   ├── cl100k-base.ts           # GPT-4, GPT-3.5 encoding
│   ├── o200k-base.ts            # GPT-4o, GPT-5 encoding
│   ├── p50k-base.ts             # Legacy Codex encoding
│   ├── claude-estimation.ts     # Claude estimation (with safety multiplier)
│   └── index.ts                 # Encoding exports
│
├── __tests__/                   # Test files
│   ├── tokenizer.test.ts        # Estimation API tests
│   └── bpe-exact.test.ts        # Exact BPE tests
│
├── tiktoken.ts                  # High-level Tiktoken class (async API)
├── tokenizer.ts                 # Sync estimation API
├── bpe.ts                       # Estimation-based BPE tokenizer
├── models.ts                    # Model configurations
├── types.ts                     # TypeScript interfaces
├── utils.ts                     # Utility functions
└── index.ts                     # Main exports
```

## Key Design Decisions

### 1. Dual API Design

- **Async API** (`getEncodingAsync`): Exact BPE, loads vocabulary from CDN
- **Sync API** (`countTokens`): Fast estimation, no network required

### 2. Claude Estimation Strategy

Claude models use a **proprietary tokenizer** that's not publicly available. We handle this with:

- Dedicated `claude_estimation` encoding
- 1.25x safety multiplier (covers 16-30% overhead vs GPT-4)
- Uses cl100k_base patterns as base (70% vocabulary overlap)

**Important**: Claude estimates intentionally over-count to prevent API truncation.

### 3. Vocabulary Loading

Vocabularies are loaded lazily from OpenAI's CDN to keep package size small (~50KB vs ~20MB with embedded vocab).

## Common Tasks

### Adding a New OpenAI Model

1. Add to `MODEL_CONFIGS` in `src/models.ts`
2. Add aliases to `MODEL_ALIASES` if needed
3. Add tests in `src/__tests__/tokenizer.test.ts`

### Adding a New Encoding

1. Create encoding file in `src/encodings/`
2. Add config to `src/bpe.ts` (`getEncodingConfig`)
3. Add definition to `src/core/encoding-definitions.ts`
4. Update `EncodingName` type in `src/types.ts`
5. Export from `src/encodings/index.ts` and `src/index.ts`
6. Add tests

### Updating Claude Estimation

The safety multiplier is defined in `src/bpe.ts`:

```typescript
export const CLAUDE_SAFETY_MULTIPLIER = 1.25;
```

If research indicates a different value is needed, update this constant.

## Code Style Guidelines

### TypeScript

- Use strict TypeScript with all checks enabled
- Prefer `const` over `let`
- Use explicit return types for public functions
- Document public APIs with JSDoc

### Naming Conventions

- **Files**: `kebab-case.ts`
- **Functions/Variables**: `camelCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Types/Interfaces**: `PascalCase`
- **Encoding names**: `snake_case` (e.g., `cl100k_base`)

### Testing

- All public APIs must have tests
- Use Vitest for testing
- Place tests in `src/__tests__/`
- Use descriptive test names

## Important Notes

### Never Break Backward Compatibility

- Existing encodings must continue to work
- Model configurations must remain stable
- Public API signatures must not change

### Claude Tokenization is Estimation Only

- No exact BPE for Claude (no public vocabulary)
- Always document that Claude counts are estimates
- Safety multiplier exists to prevent truncation

### Vocabulary URLs

Vocabulary files are loaded from OpenAI's public blob storage:

- `https://openaipublic.blob.core.windows.net/encodings/{encoding}.tiktoken`

Do NOT embed vocabularies in the package (keeps bundle small).

## Testing Claude Changes

When modifying Claude estimation:

```bash
# Run Claude-specific tests
npm test -- --grep "Claude"

# Verify safety multiplier is applied
npm test -- --grep "safety multiplier"
```

## Reference Documents

- `claude-tokenizer-research.md` - Research on Claude's tokenizer
- `ARCHITECTURE.md` - Detailed architecture documentation
- `README.md` - User-facing documentation

## Commit Guidelines

- Use conventional commits (feat:, fix:, docs:, etc.)
- Include tests with feature changes
- Update documentation when changing public APIs
- Never commit without running tests first
