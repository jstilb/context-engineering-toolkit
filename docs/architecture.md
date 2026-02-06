# Architecture

## System Overview

```mermaid
graph TD
    A[Input Text] --> B{Strategy Selection}
    B -->|Large reduction needed| C[Extractive Compression]
    B -->|Small reduction needed| D[Smart Truncation]
    B -->|Multiple sources| E[Priority Assembly]

    C --> F[Sentence Scoring]
    F --> G[Greedy Selection]
    G --> H[Compressed Text]

    D --> I[Token-Aware Split]
    I --> J[Boundary Detection]
    J --> H

    E --> K[Priority Sort]
    K --> L[Budget Enforcement]
    L --> M[Category Grouping]
    M --> H

    H --> N[Retention Benchmark]
    N --> O[Quality Report]
```

## Token Counting Pipeline

```mermaid
graph LR
    A[Text Input] --> B[Model Selection]
    B --> C[tiktoken Encoding]
    C --> D[Token Count]
    D --> E[Cost Estimation]
    D --> F[Utilization Calc]
    D --> G[Budget Tracking]
```

## Compression Decision Flow

```mermaid
flowchart TD
    A[Input: text + target_tokens] --> B{text fits in target?}
    B -->|Yes| C[Return original]
    B -->|No| D{Compression method?}
    D -->|Extractive| E[Split into sentences]
    E --> F[Score by TF-IDF + position]
    F --> G[Greedy select by score]
    G --> H[Reassemble in order]
    D -->|Truncation| I{Strategy?}
    I -->|Head| J[Keep beginning + ellipsis]
    I -->|Tail| K[Ellipsis + keep end]
    I -->|Middle| L[Keep beginning + ellipsis + keep end]
    H --> M[Output]
    J --> M
    K --> M
    L --> M
```

## Context Assembly for RAG

```mermaid
sequenceDiagram
    participant App as Application
    participant Asm as PriorityAssembler
    participant TC as TokenCounter

    App->>Asm: add(system_prompt, REQUIRED)
    App->>Asm: add(rag_chunks, HIGH)
    App->>Asm: add(chat_history, MEDIUM)
    App->>Asm: assemble()

    Asm->>TC: count(system_prompt)
    TC-->>Asm: 50 tokens
    Note over Asm: Include REQUIRED (50 used)

    Asm->>TC: count(rag_chunk_1)
    TC-->>Asm: 200 tokens
    Note over Asm: Include HIGH (250 used)

    Asm->>TC: count(rag_chunk_2)
    TC-->>Asm: 300 tokens
    Note over Asm: Exceeds budget, exclude

    Asm-->>App: AssemblyResult
```

## Design Principles

1. **Token-first thinking**: Every operation is token-aware, not character-aware
2. **Budget before build**: Plan token allocation before assembling context
3. **Measure everything**: Retention benchmarks prove compression quality
4. **Model-aware**: Different models tokenize differently and cost differently
5. **Composable**: Each component works independently or together
