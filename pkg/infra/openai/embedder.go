// Package openai はOpenAI APIのインフラ実装を提供する。
package openai

import (
	"context"
	"fmt"

	oai "github.com/sashabaranov/go-openai"
)

// Embedder はOpenAI Embedding APIの実装。domain.Embedderを満たす。
type Embedder struct {
	client *oai.Client
	model  oai.EmbeddingModel
}

func NewEmbedder(apiKey string) *Embedder {
	return &Embedder{
		client: oai.NewClient(apiKey),
		model:  oai.SmallEmbedding3,
	}
}

func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := e.client.CreateEmbeddings(ctx, oai.EmbeddingRequest{
		Model: e.model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("openai embedding: %w", err)
	}
	vecs := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}
