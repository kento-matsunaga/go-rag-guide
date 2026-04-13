// 第5章: Advanced Retrieval — 検索精度を改善する3つの手法
//
// Naive RAGの検索を3つの手法で強化し、精度の変化を比較する。
//
// 1. Baseline: 普通のベクトル検索（第3章と同じ）
// 2. Re-ranking: 検索結果をLLMで再スコアリング
// 3. HyDE: 仮想回答を生成してから検索
// 4. Multi-query: 質問を言い換えて複数回検索
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	openai "github.com/sashabaranov/go-openai"

	"github.com/kento-matsunaga/go-rag-guide/pkg/chunker"
)

// ── 基盤 ──────────────────────────────────────────

type vecChunk struct {
	chunker.Chunk
	Vector []float32
}

type result struct {
	Title      string
	Content    string
	Similarity float64
}

var client *openai.Client

func embed(texts []string) ([][]float32, error) {
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

func chat(prompt string) (string, error) {
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:     openai.GPT4oMini,
		MaxTokens: 512,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
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

func searchTopK(chunks []vecChunk, qVec []float32, k int) []result {
	rs := make([]result, len(chunks))
	for i, c := range chunks {
		rs[i] = result{c.Title, c.Content, cosineSim(qVec, c.Vector)}
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i].Similarity > rs[j].Similarity })
	if k < len(rs) {
		return rs[:k]
	}
	return rs
}

// ── 手法1: Baseline ──────────────────────────────
// 普通のベクトル検索。第3章と同じ。

func baseline(chunks []vecChunk, question string, topK int) ([]result, error) {
	qv, err := embed([]string{question})
	if err != nil {
		return nil, err
	}
	return searchTopK(chunks, qv[0], topK), nil
}

// ── 手法2: Re-ranking ──────────────────────────────
// ベクトル検索で広めに取得（top-10）し、LLMで「質問との関連度」を0-10で
// スコアリングして、上位K件に絞り込む。
//
// 本番ではCohere Rerank APIやCross-encoderを使うが、
// 仕組みを理解するためにLLMベースで実装する。

func rerank(chunks []vecChunk, question string, topK int) ([]result, error) {
	// まずベクトル検索で広めに取得
	qv, err := embed([]string{question})
	if err != nil {
		return nil, err
	}
	candidates := searchTopK(chunks, qv[0], 10)

	// LLMで各候補をスコアリング
	type scored struct {
		result
		LLMScore int
	}
	var scoredResults []scored

	for _, c := range candidates {
		prompt := fmt.Sprintf(`以下の質問と文書の関連度を0〜10の整数で評価してください。
数字のみを回答してください。

質問: %s

文書: %s`, question, c.Content)

		ans, err := chat(prompt)
		if err != nil {
			continue
		}
		score, err := strconv.Atoi(strings.TrimSpace(ans))
		if err != nil {
			score = 0
		}
		scoredResults = append(scoredResults, scored{c, score})
	}

	// LLMスコアで再ソート
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].LLMScore > scoredResults[j].LLMScore
	})

	var out []result
	for i, s := range scoredResults {
		if i >= topK {
			break
		}
		s.result.Similarity = float64(s.LLMScore) / 10.0 // スコアを0-1に正規化
		out = append(out, s.result)
	}
	return out, nil
}

// ── 手法3: HyDE ──────────────────────────────────
// ユーザーの質問に対して、LLMに「仮想的な回答」を生成させる。
// その仮想回答のEmbeddingで検索する。
//
// なぜ効くか:
//   質問文: 「有給は何日？」 ← 短くて情報が少ない
//   仮想回答: 「入社6ヶ月後に10日の有給休暇が...」 ← 文書に近い文体
//   → 仮想回答のベクトルのほうが、実際の文書のベクトルに近くなる

func hyde(chunks []vecChunk, question string, topK int) ([]result, error) {
	// LLMに仮想回答を生成させる
	prompt := fmt.Sprintf(`以下の質問に対する回答を、社内規定の文書のような文体で書いてください。
正確さは不要です。それらしい内容を1〜2文で書いてください。

質問: %s`, question)

	hypothetical, err := chat(prompt)
	if err != nil {
		return nil, err
	}

	// 仮想回答をベクトル化して検索
	hv, err := embed([]string{hypothetical})
	if err != nil {
		return nil, err
	}
	return searchTopK(chunks, hv[0], topK), nil
}

// ── 手法4: Multi-query ──────────────────────────────
// 元の質問をLLMで3つの異なる表現に言い換え、
// それぞれで検索して結果を統合する（Reciprocal Rank Fusion）。
//
// なぜ効くか:
//   元の質問: 「有給は何日？」
//   言い換え1: 「年次有給休暇の日数は？」
//   言い換え2: 「休暇の付与日数について」
//   言い換え3: 「有休は何日もらえるか」
//   → 単一の質問では見逃す文書を複数の角度から捕捉

func multiQuery(chunks []vecChunk, question string, topK int) ([]result, error) {
	// LLMで質問を3つに言い換え
	prompt := fmt.Sprintf(`以下の質問を、意味は同じだが異なる表現で3つ書き換えてください。
1行に1つずつ、番号なしで出力してください。

質問: %s`, question)

	ans, err := chat(prompt)
	if err != nil {
		return nil, err
	}

	queries := []string{question} // 元の質問も含める
	for _, line := range strings.Split(ans, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			queries = append(queries, line)
		}
	}

	// 各クエリで検索し、Reciprocal Rank Fusionでスコアを統合
	scoreMap := make(map[string]float64) // content → RRFスコア
	contentMap := make(map[string]result)
	const k = 60.0 // RRFのパラメータ

	for _, q := range queries {
		qv, err := embed([]string{q})
		if err != nil {
			continue
		}
		rs := searchTopK(chunks, qv[0], 5)
		for rank, r := range rs {
			scoreMap[r.Content] += 1.0 / (k + float64(rank+1))
			contentMap[r.Content] = r
		}
	}

	// RRFスコアで並び替え
	type rrfResult struct {
		result
		RRFScore float64
	}
	var combined []rrfResult
	for content, score := range scoreMap {
		r := contentMap[content]
		r.Similarity = score
		combined = append(combined, rrfResult{r, score})
	}
	sort.Slice(combined, func(i, j int) bool {
		return combined[i].RRFScore > combined[j].RRFScore
	})

	var out []result
	for i, c := range combined {
		if i >= topK {
			break
		}
		out = append(out, c.result)
	}
	return out, nil
}

// ── テストケース ──────────────────────────────────

type testCase struct {
	Question string
	Keyword  string // 正解を含むキーワード
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

// ── メイン ──────────────────────────────────────

func main() {
	client = openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// 文書読み込み & チャンク分割（subheading戦略）
	data, _ := os.ReadFile("cmd/ch05-advanced-retrieval/company_rules.md")
	chunks := chunker.BySubHeading(string(data))

	// ベクトル化
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}
	vecs, err := embed(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Embedding error: %v\n", err)
		os.Exit(1)
	}
	indexed := make([]vecChunk, len(chunks))
	for i, c := range chunks {
		indexed[i] = vecChunk{c, vecs[i]}
	}
	fmt.Printf("インデックス作成完了: %d チャンク\n", len(indexed))

	// ── 各手法で検索精度を比較 ──
	type strategy struct {
		Name string
		Fn   func([]vecChunk, string, int) ([]result, error)
	}
	strategies := []strategy{
		{"Baseline（ベクトル検索のみ）", baseline},
		{"Re-ranking（LLMで再スコアリング）", rerank},
		{"HyDE（仮想回答で検索）", hyde},
		{"Multi-query（質問言い換え+RRF）", multiQuery},
	}

	fmt.Println("\n================================================================")
	fmt.Println("検索精度の比較（8問、top-3）")
	fmt.Println("================================================================")

	for _, s := range strategies {
		hit := 0
		for _, tc := range tests {
			results, err := s.Fn(indexed, tc.Question, 3)
			if err != nil {
				continue
			}
			for _, r := range results {
				if strings.Contains(r.Content, tc.Keyword) {
					hit++
					break
				}
			}
		}
		pct := float64(hit) / float64(len(tests)) * 100
		bar := ""
		for i := 0; i < hit; i++ {
			bar += "✅"
		}
		for i := 0; i < len(tests)-hit; i++ {
			bar += "❌"
		}
		fmt.Printf("\n📊 %s\n   正解率: %d/%d (%.0f%%) %s\n", s.Name, hit, len(tests), pct, bar)
	}

	// ── 詳細比較: 1問だけ各手法の検索結果を表示 ──
	fmt.Println("\n================================================================")
	fmt.Println("詳細比較: 「フレックスタイムのコアタイムは？」")
	fmt.Println("================================================================")

	q := "フレックスタイムのコアタイムは？"
	for _, s := range strategies {
		results, err := s.Fn(indexed, q, 3)
		if err != nil {
			fmt.Printf("\n❌ %s: %v\n", s.Name, err)
			continue
		}
		fmt.Printf("\n📊 %s\n", s.Name)
		for i, r := range results {
			preview := r.Content
			if len([]rune(preview)) > 50 {
				preview = string([]rune(preview)[:50])
			}
			title := r.Title
			if title == "" {
				title = "(なし)"
			}
			fmt.Printf("   [%d] %.4f | %s | %s...\n", i+1, r.Similarity, title, preview)
		}
	}
}
