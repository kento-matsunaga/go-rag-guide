package domain

import "context"

// Embedder はテキストをベクトルに変換するインターフェース。
// OpenAI、Cohere、ローカルモデル等、実装を差し替え可能にする。
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// VectorStore はベクトルの保存・検索を行うインターフェース。
// インメモリ、Qdrant、pgvector等、実装を差し替え可能にする。
type VectorStore interface {
	Add(ctx context.Context, vectors []Vector) error
	Search(ctx context.Context, query []float32, topK int) ([]SearchResult, error)
}

// Generator はLLMで回答を生成するインターフェース。
// OpenAI、Claude、ローカルモデル等、実装を差し替え可能にする。
type Generator interface {
	Generate(ctx context.Context, question string, contexts []SearchResult) (string, error)
}
