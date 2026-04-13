// Package vectorstore はベクトルストアのインフラ実装を提供する。
package vectorstore

import (
	"context"
	"math"
	"sort"

	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
)

// InMemory はGoスライスによるインメモリVector Store。domain.VectorStoreを満たす。
// 学習用。本番ではQdrant等に差し替える。
type InMemory struct {
	vectors []domain.Vector
}

func NewInMemory() *InMemory {
	return &InMemory{}
}

func (s *InMemory) Add(_ context.Context, vectors []domain.Vector) error {
	s.vectors = append(s.vectors, vectors...)
	return nil
}

func (s *InMemory) Search(_ context.Context, query []float32, topK int) ([]domain.SearchResult, error) {
	results := make([]domain.SearchResult, len(s.vectors))
	for i, v := range s.vectors {
		results[i] = domain.SearchResult{
			Chunk: v.Chunk,
			Score: cosineSim(query, v.Values),
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if topK < len(results) {
		return results[:topK], nil
	}
	return results, nil
}

func cosineSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
