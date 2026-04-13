// 第7章: RAGを定量評価する
//
// 4つの評価指標でRAGパイプラインの品質を測定し、
// パラメータ（チャンキング戦略 × top-K）を変えて精度変化を比較する。
package main

import (
	"context"
	"fmt"
	"os"

	oai "github.com/sashabaranov/go-openai"

	"github.com/kento-matsunaga/go-rag-guide/pkg/chunker"
	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
	"github.com/kento-matsunaga/go-rag-guide/pkg/eval"
	oaiinfra "github.com/kento-matsunaga/go-rag-guide/pkg/infra/openai"
	"github.com/kento-matsunaga/go-rag-guide/pkg/infra/vectorstore"
	"github.com/kento-matsunaga/go-rag-guide/pkg/usecase"
)

// LLMJudge — GPT-4o-miniで評価を行う
type llmJudge struct {
	client *oai.Client
}

func (j *llmJudge) Judge(ctx context.Context, prompt string) (string, error) {
	resp, err := j.client.CreateChatCompletion(ctx, oai.ChatCompletionRequest{
		Model:     oai.GPT4oMini,
		MaxTokens: 16,
		Messages: []oai.ChatCompletionMessage{
			{Role: oai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

// テストデータセット — 質問と期待されるキーワード
var testCases = []eval.TestCase{
	{"有給休暇は何日もらえますか？", "10日"},
	{"リモートワークは週何日まで可能？", "週3日"},
	{"AWS認定の技術手当はいくら？", "20,000円"},
	{"シニアエンジニアの昇進要件は？", "2期連続"},
	{"書籍購入の補助額は？", "5,000円"},
	{"フレックスタイムのコアタイムは？", "午前11時"},
	{"残業手当は通常賃金の何%増？", "25%"},
	{"リフレッシュ休暇は何日間？", "5日間"},
}

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")
	judge := &llmJudge{client: oai.NewClient(apiKey)}

	data, _ := os.ReadFile("cmd/ch07-evaluation/company_rules.md")
	doc := string(data)

	// ── パラメータスイープ: チャンキング戦略 × top-K ──
	type config struct {
		Name   string
		Chunks []chunker.Chunk
		TopK   int
	}

	configs := []config{
		{"heading × top-3", chunker.ByHeading(doc), 3},
		{"heading × top-1", chunker.ByHeading(doc), 1},
		{"subheading × top-3", chunker.BySubHeading(doc), 3},
		{"subheading × top-1", chunker.BySubHeading(doc), 1},
		{"fixed-400 × top-3", chunker.ByFixedSize(doc, 400, 100), 3},
	}

	fmt.Println("================================================================")
	fmt.Println("RAG評価パイプライン — パラメータスイープ")
	fmt.Println("================================================================")
	fmt.Println()
	fmt.Printf("テストケース: %d問\n", len(testCases))
	fmt.Printf("評価指標: Hit@K, Faithfulness, Answer Relevancy, Correctness\n")

	for _, cfg := range configs {
		fmt.Printf("\n────────────────────────────────────────\n")
		fmt.Printf("📦 %s（%dチャンク）\n", cfg.Name, len(cfg.Chunks))
		fmt.Printf("────────────────────────────────────────\n")

		// domainチャンクに変換
		domainChunks := make([]domain.Chunk, len(cfg.Chunks))
		for i, c := range cfg.Chunks {
			domainChunks[i] = domain.Chunk{
				ID:       fmt.Sprintf("c%d", i),
				Content:  c.Content,
				Metadata: map[string]string{"title": c.Title},
			}
		}

		// RAGパイプライン構築
		embedder := oaiinfra.NewEmbedder(apiKey)
		store := vectorstore.NewInMemory()
		gen := oaiinfra.NewGenerator(apiKey)
		rag := usecase.New(embedder, store, gen)

		if err := rag.Index(ctx, domainChunks); err != nil {
			fmt.Printf("  ❌ Index error: %v\n", err)
			continue
		}

		// 各テストケースを評価
		var results []eval.Result
		for _, tc := range testCases {
			answer, searchResults, err := rag.Ask(ctx, tc.Question, cfg.TopK)
			if err != nil {
				fmt.Printf("  ❌ %s: %v\n", tc.Question, err)
				continue
			}

			// Hit@K
			hit := eval.HitAtK(searchResults, tc.GroundTruth)

			// Faithfulness（LLM-as-judge）
			faith, _ := eval.Faithfulness(ctx, judge, tc.Question, answer, searchResults)

			// Answer Relevancy（LLM-as-judge）
			rel, _ := eval.AnswerRelevancy(ctx, judge, tc.Question, answer)

			// Correctness（キーワード一致）
			correct := eval.Correctness(answer, tc.GroundTruth)

			results = append(results, eval.Result{
				Question:        tc.Question,
				Answer:          answer,
				GroundTruth:     tc.GroundTruth,
				HitAtK:          hit,
				Faithfulness:    faith,
				AnswerRelevancy: rel,
				Correctness:     correct,
			})

			// 個別結果
			hitMark := "✅"
			if !hit {
				hitMark = "❌"
			}
			fmt.Printf("  %s %.1f %.1f %.1f | %s\n",
				hitMark, faith, rel, correct, tc.Question)
		}

		// サマリー
		report := eval.Summarize(results)
		fmt.Printf("\n  ┌─────────────────────────────────┐\n")
		fmt.Printf("  │ Hit@K:       %.0f%%                │\n", report.AvgHitRate*100)
		fmt.Printf("  │ Faithfulness: %.2f              │\n", report.AvgFaithful)
		fmt.Printf("  │ Relevancy:    %.2f              │\n", report.AvgRelevancy)
		fmt.Printf("  │ Correctness:  %.0f%%               │\n", report.AvgCorrect*100)
		fmt.Printf("  └─────────────────────────────────┘\n")
	}
}
