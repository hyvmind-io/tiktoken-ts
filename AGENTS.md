# AGENTS.md - AI Agent Guide for tiktoken-ts

This file provides guidance for AI agents working with the tiktoken-ts codebase.

## Repository Overview

tiktoken-ts is a pure TypeScript implementation of OpenAI's tiktoken tokenizer, providing:

- **Exact BPE tokenization** (async, requires vocabulary loading from CDN)
- **Fast estimation API** (sync, heuristic-based, no network required)
- **Claude-specific estimation** with safety multiplier for Anthropic models
- **Model configuration** for context limits and encoding selection

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
