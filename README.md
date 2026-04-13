# Go RAG Guide — サンプルコード

Zenn Book「**GoエンジニアのためのRAG実践入門**」のサンプルコードリポジトリです。

## セットアップ

```bash
git clone https://github.com/kento-matsunaga/go-rag-guide.git
cd go-rag-guide
go mod tidy
export OPENAI_API_KEY="your-api-key"
```

OpenAI APIキーは https://platform.openai.com/api-keys で発行してください。

## 各章のコード

| 章 | ディレクトリ | 実行コマンド |
|----|------------|-------------|
| 第3章 Embedding体験 | `cmd/ch03-embedding/` | `go run ./cmd/ch03-embedding/` |
| 第3章 Naive RAG | `cmd/ch03-naive-rag/` | `go run ./cmd/ch03-naive-rag/` |
| 第4章 チャンキング比較 | `cmd/ch04-chunking/` | `go run ./cmd/ch04-chunking/` |
| 第5章 検索改善 | `cmd/ch05-advanced-retrieval/` | `go run ./cmd/ch05-advanced-retrieval/` |
| 第6章 Clean Architecture | `cmd/ch06-clean-rag/` | `go run ./cmd/ch06-clean-rag/` |
| 第7章 評価パイプライン | `cmd/ch07-evaluation/` | `go run ./cmd/ch07-evaluation/` |
| 第8章 REST API | `cmd/ch08-api/` | `go run ./cmd/ch08-api/` |

## 共通パッケージ

| パッケージ | 説明 |
|-----------|------|
| `pkg/domain/` | ドメインモデル（Chunk, Vector, SearchResult）とインターフェース |
| `pkg/usecase/` | RAGパイプライン（Index, Ask） |
| `pkg/infra/openai/` | OpenAI API実装（Embedder, Generator） |
| `pkg/infra/vectorstore/` | InMemory Vector Store |
| `pkg/chunker/` | チャンキング戦略（Heading, SubHeading, FixedSize） |
| `pkg/eval/` | 評価指標（Hit@K, Faithfulness, Relevancy, Correctness） |

## 必要なもの

- Go 1.22+
- OpenAI APIキー（Embedding: text-embedding-3-small, LLM: GPT-4o-mini）

## ライセンス

MIT
