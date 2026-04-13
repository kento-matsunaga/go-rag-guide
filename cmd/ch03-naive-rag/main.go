// GoでRAGをゼロから実装する
//
// 外部のVector DBもフレームワークも使わない。
// Goのスライスだけで、RAGの全工程を自前実装する。
//
// 全体の流れ:
//   1. 文書を読み込む
//   2. チャンクに分割する
//   3. 各チャンクをベクトルに変換する（OpenAI Embedding API）
//   4. ユーザーの質問をベクトル化し、類似チャンクを検索する
//   5. 検索結果をLLMに渡して回答を生成する（GPT-4o-mini）
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// データ構造
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Chunk — 文書を分割した断片。RAGの最小検索単位。
type Chunk struct {
	Title   string    // セクション名（メタデータ）
	Content string    // テキスト本文
	Vector  []float32 // Embeddingベクトル（1536次元）
}

// SearchResult — 検索結果。チャンクと類似度のペア。
type SearchResult struct {
	Chunk      Chunk
	Similarity float64
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 1: 文書を読み込む
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

func loadDocument(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	return string(data), nil
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 2: チャンクに分割する
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Markdownの「## 」見出しで分割する。最もシンプルな構造ベース分割。

func splitByHeading(doc string) []Chunk {
	var chunks []Chunk
	var title, content string

	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "## ") {
			// 前のセクションを保存
			if strings.TrimSpace(content) != "" {
				chunks = append(chunks, Chunk{
					Title:   title,
					Content: strings.TrimSpace(content),
				})
			}
			title = strings.TrimPrefix(line, "## ")
			content = ""
		} else {
			content += line + "\n"
		}
	}
	// 最後のセクションも保存
	if strings.TrimSpace(content) != "" {
		chunks = append(chunks, Chunk{
			Title:   title,
			Content: strings.TrimSpace(content),
		})
	}
	return chunks
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 3: Embeddingベクトルを取得する
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 4: 類似検索（自前Vector Store）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Vector DBを使わず、Goのスライスで実装する。
// 全チャンクとのコサイン類似度を計算し、上位K件を返す。
//
// 本番ではQdrant等のVector DBを使うが、まず自前で仕組みを理解する。

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

func search(chunks []Chunk, queryVec []float32, topK int) []SearchResult {
	results := make([]SearchResult, len(chunks))
	for i, c := range chunks {
		results[i] = SearchResult{
			Chunk:      c,
			Similarity: cosineSim(queryVec, c.Vector),
		}
	}
	// 類似度が高い順にソート
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if topK < len(results) {
		return results[:topK]
	}
	return results
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 5: LLMで回答を生成する
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

func generate(client *openai.Client, question string, results []SearchResult) (string, error) {
	// 検索結果のテキストを結合してコンテキストを作る
	var parts []string
	for _, r := range results {
		parts = append(parts, r.Chunk.Content)
	}
	ragContext := strings.Join(parts, "\n\n---\n\n")

	// LLMに渡すプロンプトを組み立てる
	prompt := fmt.Sprintf(`以下の社内規定の情報を参考に、質問に正確に答えてください。
情報に含まれていない内容については「その情報は見つかりませんでした」と回答してください。

【参考情報】
%s

【質問】
%s`, ragContext, question)

	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:     openai.GPT4oMini, // 安くて十分な性能
			MaxTokens: 1024,
			Messages: []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleUser, Content: prompt},
			},
		},
	)
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// メイン: すべてをつなげる
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// ── STEP 1: 文書を読み込む ──
	fmt.Println("STEP 1: 文書を読み込む")
	doc, err := loadDocument("cmd/ch03-naive-rag/company_rules.md")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  %d文字読み込み完了\n", len(doc))

	// ── STEP 2: チャンクに分割 ──
	fmt.Println("\nSTEP 2: チャンクに分割する")
	chunks := splitByHeading(doc)
	fmt.Printf("  %d個のチャンクに分割\n", len(chunks))
	for i, c := range chunks {
		fmt.Printf("  [%d] %s（%d文字）\n", i, c.Title, len(c.Content))
	}

	// ── STEP 3: ベクトルに変換 ──
	fmt.Println("\nSTEP 3: 各チャンクをベクトルに変換する")
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}
	vecs, err := embed(client, texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Embedding error: %v\n", err)
		os.Exit(1)
	}
	for i := range chunks {
		chunks[i].Vector = vecs[i]
	}
	fmt.Printf("  %d個のベクトル（各%d次元）を生成\n", len(vecs), len(vecs[0]))

	// ── STEP 4 & 5: 質問 → 検索 → 回答 ──
	questions := []string{
		"有給休暇は何日もらえますか？",
		"リモートワークのルールを教えて",
		"AWS資格を取ると手当はいくら？",
		"シニアエンジニアになるには？",
	}

	fmt.Println("\n============================================================")
	fmt.Println("STEP 4 & 5: 質問 → 検索 → 回答生成")
	fmt.Println("============================================================")

	for _, q := range questions {
		fmt.Printf("\n──────────────────────────────────────\n")
		fmt.Printf("❓ 質問: %s\n", q)

		// 質問をベクトル化
		qVecs, err := embed(client, []string{q})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Query embedding error: %v\n", err)
			continue
		}

		// 類似検索（上位3件）
		results := search(chunks, qVecs[0], 3)
		fmt.Println("\n  📎 検索結果:")
		for i, r := range results {
			fmt.Printf("     [%d] %s (類似度: %.4f)\n", i+1, r.Chunk.Title, r.Similarity)
		}

		// LLMで回答生成
		answer, err := generate(client, q, results)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Generation error: %v\n", err)
			continue
		}
		fmt.Printf("\n  💬 回答:\n  %s\n", answer)
	}
}
