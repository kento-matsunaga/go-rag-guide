// Step 1: GoでEmbeddingを体験する
//
// テキストをベクトルに変換し、意味の近さ＝ベクトルの近さを確認する。
package main

import (
	"context"
	"fmt"
	"math"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// ── Embedding取得 ──────────────────────────────────
// OpenAI APIにテキストを送り、1536次元のベクトルを受け取る。
func getEmbeddings(client *openai.Client, texts []string) ([][]float32, error) {
	resp, err := client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Model: openai.SmallEmbedding3, // text-embedding-3-small（1536次元）
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("embedding error: %w", err)
	}

	result := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		result[i] = d.Embedding
	}
	return result, nil
}

// ── コサイン類似度 ──────────────────────────────────
// 2つのベクトルがどれくらい同じ方向を向いているか。
// 1.0 = 完全に同じ意味、0.0 = 無関係、-1.0 = 正反対
func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func main() {
	// APIキーは環境変数から取得（コードに書かない）
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// ── 5つの文を用意 ──
	sentences := []string{
		"Goでサーバーを実装する",        // A: プログラミング
		"Golangでバックエンドを開発する", // B: Aとほぼ同じ意味
		"今日の東京の天気は晴れです",     // C: 全く別の話題
		"機械学習モデルをトレーニングする", // D: プログラミング寄り
		"明日は雨が降るらしい",          // E: 天気の話題
	}

	// ── ベクトルに変換 ──
	embeddings, err := getEmbeddings(client, sentences)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("============================================================")
	fmt.Println("GoでEmbeddingを体験する")
	fmt.Println("============================================================")

	for i, s := range sentences {
		fmt.Printf("\n「%s」\n", s)
		fmt.Printf("  → %d次元のベクトル\n", len(embeddings[i]))
		fmt.Printf("  → 先頭5要素: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			embeddings[i][0], embeddings[i][1], embeddings[i][2],
			embeddings[i][3], embeddings[i][4])
	}

	// ── 類似度を計算 ──
	fmt.Println("\n============================================================")
	fmt.Println("コサイン類似度（1に近いほど意味が近い）")
	fmt.Println("============================================================")

	pairs := []struct {
		i, j  int
		label string
	}{
		{0, 1, "同じ意味（Goサーバー vs Golangバックエンド）"},
		{0, 3, "関連あり（Goサーバー vs ML訓練）"},
		{0, 2, "無関係（Goサーバー vs 天気）"},
		{2, 4, "同じ話題（天気 vs 天気）"},
		{1, 4, "無関係（Golang vs 天気）"},
	}

	for _, p := range pairs {
		sim := cosineSimilarity(embeddings[p.i], embeddings[p.j])
		bar := ""
		for k := 0; k < int(sim*30); k++ {
			bar += "█"
		}
		fmt.Printf("\n  %s\n", p.label)
		fmt.Printf("  %.4f %s\n", sim, bar)
	}
}
