package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
	oai "github.com/sashabaranov/go-openai"
)

// Generator はOpenAI Chat Completion APIの実装。domain.Generatorを満たす。
type Generator struct {
	client *oai.Client
	model  string
}

func NewGenerator(apiKey string) *Generator {
	return &Generator{
		client: oai.NewClient(apiKey),
		model:  oai.GPT4oMini,
	}
}

func (g *Generator) Generate(ctx context.Context, question string, results []domain.SearchResult) (string, error) {
	var parts []string
	for _, r := range results {
		parts = append(parts, r.Chunk.Content)
	}
	ragCtx := strings.Join(parts, "\n\n---\n\n")

	prompt := fmt.Sprintf(`以下の情報を参考に、質問に正確に答えてください。
情報に含まれていない内容は「その情報は見つかりませんでした」と答えてください。

【参考情報】
%s

【質問】
%s`, ragCtx, question)

	resp, err := g.client.CreateChatCompletion(ctx, oai.ChatCompletionRequest{
		Model:     g.model,
		MaxTokens: 1024,
		Messages: []oai.ChatCompletionMessage{
			{Role: oai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", fmt.Errorf("openai chat: %w", err)
	}
	return resp.Choices[0].Message.Content, nil
}
