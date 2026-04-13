// 第8章: RAGをREST APIとして公開する
//
// 第6章のClean Architecture RAGを、Echo WebフレームワークでHTTP API化する。
// 既存のGoバックエンドにRAG機能を組み込むパターン。
//
// エンドポイント:
//   POST /api/ask     — 質問を受け取り、RAGで回答を返す
//   POST /api/index   — 文書をインデックスに追加する
//   GET  /api/health  — ヘルスチェック
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"

	"github.com/kento-matsunaga/go-rag-guide/pkg/chunker"
	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
	oaiinfra "github.com/kento-matsunaga/go-rag-guide/pkg/infra/openai"
	"github.com/kento-matsunaga/go-rag-guide/pkg/infra/vectorstore"
	"github.com/kento-matsunaga/go-rag-guide/pkg/usecase"
)

// ── リクエスト/レスポンス ──

type AskRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k,omitempty"` // デフォルト3
}

type AskResponse struct {
	Answer  string         `json:"answer"`
	Sources []SourceResult `json:"sources"`
}

type SourceResult struct {
	Title string  `json:"title"`
	Score float64 `json:"score"`
	Text  string  `json:"text"`
}

type IndexRequest struct {
	Document string `json:"document"` // Markdownテキスト
}

type IndexResponse struct {
	ChunkCount int `json:"chunk_count"`
}

// ── ハンドラ ──

type Handler struct {
	rag *usecase.RAG
}

func (h *Handler) Ask(c echo.Context) error {
	var req AskRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "invalid request"})
	}
	if req.Question == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "question is required"})
	}
	if req.TopK <= 0 {
		req.TopK = 3
	}

	answer, results, err := h.rag.Ask(context.Background(), req.Question, req.TopK)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}

	var sources []SourceResult
	for _, r := range results {
		title := r.Chunk.Metadata["title"]
		text := r.Chunk.Content
		if len([]rune(text)) > 100 {
			text = string([]rune(text)[:100]) + "..."
		}
		sources = append(sources, SourceResult{
			Title: title,
			Score: r.Score,
			Text:  text,
		})
	}

	return c.JSON(http.StatusOK, AskResponse{
		Answer:  answer,
		Sources: sources,
	})
}

func (h *Handler) Index(c echo.Context) error {
	var req IndexRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "invalid request"})
	}
	if req.Document == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "document is required"})
	}

	rawChunks := chunker.BySubHeading(req.Document)
	chunks := make([]domain.Chunk, len(rawChunks))
	for i, rc := range rawChunks {
		chunks[i] = domain.Chunk{
			ID:       fmt.Sprintf("chunk_%d", i),
			Content:  rc.Content,
			Metadata: map[string]string{"title": rc.Title},
		}
	}

	if err := h.rag.Index(context.Background(), chunks); err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}

	return c.JSON(http.StatusOK, IndexResponse{ChunkCount: len(chunks)})
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")

	// ── DI: Clean Architectureの組み立て ──
	embedder := oaiinfra.NewEmbedder(apiKey)
	store := vectorstore.NewInMemory()
	gen := oaiinfra.NewGenerator(apiKey)
	rag := usecase.New(embedder, store, gen)

	// ── 初期文書のインデックス ──
	data, err := os.ReadFile("cmd/ch08-api/company_rules.md")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading document: %v\n", err)
		os.Exit(1)
	}
	rawChunks := chunker.BySubHeading(string(data))
	chunks := make([]domain.Chunk, len(rawChunks))
	for i, rc := range rawChunks {
		chunks[i] = domain.Chunk{
			ID:       fmt.Sprintf("init_%d", i),
			Content:  rc.Content,
			Metadata: map[string]string{"title": rc.Title},
		}
	}
	if err := rag.Index(context.Background(), chunks); err != nil {
		fmt.Fprintf(os.Stderr, "Index error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("📦 初期インデックス: %d チャンク\n", len(chunks))

	// ── Echo サーバー ──
	handler := &Handler{rag: rag}

	e := echo.New()
	e.Use(middleware.Logger())
	e.Use(middleware.CORS())

	e.GET("/api/health", func(c echo.Context) error {
		return c.JSON(http.StatusOK, map[string]string{"status": "ok"})
	})
	e.POST("/api/ask", handler.Ask)
	e.POST("/api/index", handler.Index)

	fmt.Println("🚀 RAG API server starting on :8080")
	e.Logger.Fatal(e.Start(":8080"))
}
