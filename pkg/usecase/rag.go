// Package usecase はRAGパイプラインのユースケースを実装する。
//
// Domain層のインターフェースに依存し、具体的な実装には依存しない。
// 「何のAPIを使うか」「どのDBに保存するか」はここでは知らない。
package usecase

import (
	"context"
	"fmt"

	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
)

// RAG はRAGパイプラインのユースケース。
type RAG struct {
	embedder Embedder
	store    VectorStore
	gen      Generator
}

// インターフェースをusecase内で再定義（domain.portと同一だが、依存方向を明確にする）
type Embedder = domain.Embedder
type VectorStore = domain.VectorStore
type Generator = domain.Generator

// New はRAGユースケースを生成する。
func New(e Embedder, s VectorStore, g Generator) *RAG {
	return &RAG{embedder: e, store: s, gen: g}
}

// Index はチャンク群をベクトル化してストアに保存する。
// 事前準備として一度だけ実行する。
func (r *RAG) Index(ctx context.Context, chunks []domain.Chunk) error {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}

	vecs, err := r.embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}

	vectors := make([]domain.Vector, len(chunks))
	for i, c := range chunks {
		vectors[i] = domain.Vector{Chunk: c, Values: vecs[i]}
	}

	if err := r.store.Add(ctx, vectors); err != nil {
		return fmt.Errorf("store add: %w", err)
	}
	return nil
}

// Ask はユーザーの質問に対してRAGで回答する。
// 質問 → ベクトル化 → 検索 → LLMで回答生成。
func (r *RAG) Ask(ctx context.Context, question string, topK int) (string, []domain.SearchResult, error) {
	// 質問をベクトル化
	qVecs, err := r.embedder.Embed(ctx, []string{question})
	if err != nil {
		return "", nil, fmt.Errorf("embed query: %w", err)
	}

	// 類似検索
	results, err := r.store.Search(ctx, qVecs[0], topK)
	if err != nil {
		return "", nil, fmt.Errorf("search: %w", err)
	}

	// LLMで回答生成
	answer, err := r.gen.Generate(ctx, question, results)
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}

	return answer, results, nil
}
