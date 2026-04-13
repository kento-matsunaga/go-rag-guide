// 第6章: Clean ArchitectureでRAGを実装する
//
// 第3章の1ファイルRAGを、3層のClean Architectureにリファクタリングした。
// mainの役割は「組み立て」だけ。各層は独立してテスト・差し替え可能。
//
// 構造:
//   domain/     — モデルとインターフェース（外部依存なし）
//   usecase/    — RAGパイプラインのロジック（interfaceに依存）
//   infra/      — 具体的な実装（OpenAI API、InMemory Store）
//   cmd/main.go — 組み立て（DI: Dependency Injection）
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/kento-matsunaga/go-rag-guide/pkg/chunker"
	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
	oaiinfra "github.com/kento-matsunaga/go-rag-guide/pkg/infra/openai"
	"github.com/kento-matsunaga/go-rag-guide/pkg/infra/vectorstore"
	"github.com/kento-matsunaga/go-rag-guide/pkg/usecase"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")

	// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	// DI: 具体的な実装をインターフェースに注入
	// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	// ここを差し替えるだけで、Embedderやストアを変更できる。
	// 例: oaiinfra.NewEmbedder → cohere.NewEmbedder
	// 例: vectorstore.NewInMemory → qdrant.NewClient
	embedder := oaiinfra.NewEmbedder(apiKey)
	store := vectorstore.NewInMemory()
	generator := oaiinfra.NewGenerator(apiKey)

	rag := usecase.New(embedder, store, generator)

	// ── 文書読み込み & チャンク分割 ──
	data, err := os.ReadFile("cmd/ch06-clean-rag/company_rules.md")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	rawChunks := chunker.BySubHeading(string(data))
	chunks := make([]domain.Chunk, len(rawChunks))
	for i, c := range rawChunks {
		chunks[i] = domain.Chunk{
			ID:      fmt.Sprintf("chunk_%d", i),
			Content: c.Content,
			Metadata: map[string]string{
				"title": c.Title,
			},
		}
	}

	// ── インデックス作成 ──
	fmt.Println("📦 インデックス作成中...")
	if err := rag.Index(ctx, chunks); err != nil {
		fmt.Fprintf(os.Stderr, "Index error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("   %d チャンクをインデックス化\n", len(chunks))

	// ── 質問 → 回答 ──
	questions := []string{
		"有給休暇は何日もらえますか？",
		"リモートワークのルールを教えて",
		"AWS資格を取ると手当はいくら？",
		"シニアエンジニアになるには？",
	}

	fmt.Println("\n============================================================")
	fmt.Println("Clean Architecture RAG — 質問応答")
	fmt.Println("============================================================")

	for _, q := range questions {
		fmt.Printf("\n──────────────────────────────────────\n")
		fmt.Printf("❓ %s\n", q)

		answer, results, err := rag.Ask(ctx, q, 3)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		fmt.Println("  📎 検索結果:")
		for i, r := range results {
			title := r.Chunk.Metadata["title"]
			if title == "" {
				title = "(なし)"
			}
			fmt.Printf("     [%d] %.4f | %s\n", i+1, r.Score, title)
		}
		fmt.Printf("\n  💬 %s\n", strings.ReplaceAll(answer, "\n", "\n  "))
	}
}
