// 第4章: チャンキング戦略の比較実験
//
// 3つの分割戦略で同じ文書を分割し、同じ質問で検索精度を比較する。
// 「チャンクの切り方を変えるだけで、RAGの精度がどう変わるか」を体感する。
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	openai "github.com/sashabaranov/go-openai"

	"github.com/kento-matsunaga/go-rag-guide/pkg/chunker"
)

// ── ベクトル操作 ──────────────────────────────────

func embed(client *openai.Client, texts []string) ([][]float32, error) {
	resp, err := client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Model: openai.SmallEmbedding3,
		Input: texts,
	})
	if err != nil {
		return nil, err
	}
	vecs := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
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

// ── 検索 ──────────────────────────────────────────

type indexedChunk struct {
	chunker.Chunk
	Vector []float32
}

type searchResult struct {
	Title      string
	Content    string
	Similarity float64
}

func searchTopK(chunks []indexedChunk, queryVec []float32, k int) []searchResult {
	results := make([]searchResult, len(chunks))
	for i, c := range chunks {
		results[i] = searchResult{
			Title:      c.Title,
			Content:    c.Content,
			Similarity: cosineSim(queryVec, c.Vector),
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if k < len(results) {
		return results[:k]
	}
	return results
}

// ── インデックス作成 ──────────────────────────────

func buildIndex(client *openai.Client, chunks []chunker.Chunk) ([]indexedChunk, error) {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}
	vecs, err := embed(client, texts)
	if err != nil {
		return nil, err
	}
	indexed := make([]indexedChunk, len(chunks))
	for i, c := range chunks {
		indexed[i] = indexedChunk{Chunk: c, Vector: vecs[i]}
	}
	return indexed, nil
}

// ── テストケース ──────────────────────────────────
// 各質問に「正解を含むキーワード」を設定し、検索結果に含まれるか判定する

type testCase struct {
	Question    string
	ExpectInTop string // 検索結果上位にこのキーワードが含まれれば正解
}

var tests = []testCase{
	{"有給休暇は何日もらえますか？", "10日"},
	{"リモートワークは週何日まで？", "週3日"},
	{"AWS資格の手当はいくら？", "20,000円"},
	{"シニアエンジニアの昇進要件は？", "2期連続"},
	{"書籍購入の補助額は？", "5,000円"},
	{"フレックスタイムのコアタイムは？", "午前11時"},
	{"残業手当は何%増？", "25%"},
	{"リフレッシュ休暇は何日？", "5日間"},
}

func evaluate(chunks []indexedChunk, client *openai.Client, topK int) (int, int) {
	hit, total := 0, len(tests)

	for _, tc := range tests {
		qVecs, err := embed(client, []string{tc.Question})
		if err != nil {
			continue
		}
		results := searchTopK(chunks, qVecs[0], topK)

		// 上位K件のコンテンツに正解キーワードが含まれるかチェック
		found := false
		for _, r := range results {
			if strings.Contains(r.Content, tc.ExpectInTop) {
				found = true
				break
			}
		}
		if found {
			hit++
		}
	}
	return hit, total
}

// ── メイン ──────────────────────────────────────

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// 文書読み込み
	data, err := os.ReadFile("cmd/ch04-chunking/company_rules.md")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	doc := string(data)

	// ── 3つの戦略で分割 ──
	strategies := []struct {
		Name   string
		Chunks []chunker.Chunk
	}{
		{"heading（## 見出しで分割）", chunker.ByHeading(doc)},
		{"subheading（### 条文ごとに分割）", chunker.BySubHeading(doc)},
		{"fixed-400（400文字固定、overlap 100）", chunker.ByFixedSize(doc, 400, 100)},
	}

	fmt.Println("================================================================")
	fmt.Println("チャンキング戦略の比較実験")
	fmt.Println("================================================================")

	for _, s := range strategies {
		count, avg, min, max := chunker.Stats(s.Chunks)
		fmt.Printf("\n📦 戦略: %s\n", s.Name)
		fmt.Printf("   チャンク数: %d | 平均: %.0f文字 | 最小: %d文字 | 最大: %d文字\n",
			count, avg, min, max)

		// チャンク一覧を表示
		for i, c := range s.Chunks {
			title := c.Title
			if title == "" {
				title = "(なし)"
			}
			preview := c.Content
			if len([]rune(preview)) > 40 {
				preview = string([]rune(preview)[:40])
			}
			fmt.Printf("   [%d] %s | %s...\n", i, title, preview)
		}
	}

	// ── 各戦略でインデックスを作成し、検索精度を比較 ──
	fmt.Println("\n================================================================")
	fmt.Println("検索精度の比較（8問のテストケース、top-3で評価）")
	fmt.Println("================================================================")

	for _, s := range strategies {
		indexed, err := buildIndex(client, s.Chunks)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Index error for %s: %v\n", s.Name, err)
			continue
		}

		hit, total := evaluate(indexed, client, 3)
		pct := float64(hit) / float64(total) * 100

		bar := ""
		for i := 0; i < hit; i++ {
			bar += "✅"
		}
		for i := 0; i < total-hit; i++ {
			bar += "❌"
		}

		fmt.Printf("\n📊 %s\n", s.Name)
		fmt.Printf("   正解率: %d/%d (%.0f%%) %s\n", hit, total, pct, bar)
	}

	// ── 詳細表示: 各戦略の検索結果を1問だけ見せる ──
	fmt.Println("\n================================================================")
	fmt.Println("詳細比較: 「フレックスタイムのコアタイムは？」")
	fmt.Println("================================================================")

	question := "フレックスタイムのコアタイムは？"
	qVecs, _ := embed(client, []string{question})

	for _, s := range strategies {
		indexed, _ := buildIndex(client, s.Chunks)
		results := searchTopK(indexed, qVecs[0], 3)

		fmt.Printf("\n📦 %s\n", s.Name)
		for i, r := range results {
			preview := r.Content
			if len([]rune(preview)) > 60 {
				preview = string([]rune(preview)[:60])
			}
			title := r.Title
			if title == "" {
				title = "(なし)"
			}
			fmt.Printf("   [%d] %.4f | %s | %s...\n", i+1, r.Similarity, title, preview)
		}
	}
}
