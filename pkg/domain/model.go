// Package domain はRAGパイプラインのドメインモデルを定義する。
//
// Clean Architectureの最内層。外部ライブラリに一切依存しない。
// ここで定義した型がアプリケーション全体の共通言語になる。
package domain

// Chunk は文書を分割した断片。RAGの最小検索単位。
type Chunk struct {
	ID       string            // 一意な識別子
	Content  string            // テキスト本文
	Metadata map[string]string // 付加情報（ファイル名、セクション名等）
}

// Vector はEmbeddingベクトルとチャンクのペア。
type Vector struct {
	Chunk  Chunk
	Values []float32 // Embeddingベクトル（例: 1536次元）
}

// SearchResult は検索結果。チャンクとスコアのペア。
type SearchResult struct {
	Chunk Chunk
	Score float64 // 類似度（0〜1、高いほど関連性が高い）
}
