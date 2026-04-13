// Package eval はRAGパイプラインの評価指標を実装する。
//
// RAGASフレームワーク（Python）の主要指標をGoで再実装。
// LLM-as-a-judge パターンで、LLM自身に品質を評価させる。
package eval

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/kento-matsunaga/go-rag-guide/pkg/domain"
)

// LLMJudge はLLMを使って評価を行うインターフェース。
type LLMJudge interface {
	Judge(ctx context.Context, prompt string) (string, error)
}

// TestCase はRAG評価のテストケース。
type TestCase struct {
	Question    string // 質問
	GroundTruth string // 正解（期待される回答に含まれるべきキーワードまたは文）
}

// Result は1つのテストケースの評価結果。
type Result struct {
	Question       string
	Answer         string
	GroundTruth    string
	HitAtK         bool    // 検索結果top-Kに正解チャンクが含まれるか
	Faithfulness   float64 // 回答がcontextに基づいているか（0-1）
	AnswerRelevancy float64 // 回答が質問に関連しているか（0-1）
	Correctness    float64 // 回答が正解を含んでいるか（0-1）
}

// Report は全テストケースの評価レポート。
type Report struct {
	Results       []Result
	AvgHitRate    float64
	AvgFaithful   float64
	AvgRelevancy  float64
	AvgCorrect    float64
}

// ── Hit@K: 検索結果にground truthが含まれるか ──

func HitAtK(results []domain.SearchResult, groundTruth string) bool {
	for _, r := range results {
		if strings.Contains(r.Chunk.Content, groundTruth) {
			return true
		}
	}
	return false
}

// ── Faithfulness: 回答がcontextに基づいているか ──
// RAGASの手法: 回答中の各claimがcontextで裏付けられているかをLLMが判定。
// 簡易版として、0-10スコアで一括評価する。

func Faithfulness(ctx context.Context, judge LLMJudge, question, answer string, results []domain.SearchResult) (float64, error) {
	var contextTexts []string
	for _, r := range results {
		contextTexts = append(contextTexts, r.Chunk.Content)
	}
	ctxText := strings.Join(contextTexts, "\n---\n")

	prompt := fmt.Sprintf(`以下の「回答」が「参考情報」に基づいているか評価してください。
参考情報に書かれていない内容を回答に含んでいる場合は低スコアにしてください。

0（参考情報と無関係）〜 10（完全に参考情報に基づいている）の整数で回答してください。
数字のみを出力してください。

【参考情報】
%s

【質問】
%s

【回答】
%s`, ctxText, question, answer)

	resp, err := judge.Judge(ctx, prompt)
	if err != nil {
		return 0, err
	}
	score, err := strconv.Atoi(strings.TrimSpace(resp))
	if err != nil {
		return 0, nil
	}
	return float64(score) / 10.0, nil
}

// ── Answer Relevancy: 回答が質問に関連しているか ──

func AnswerRelevancy(ctx context.Context, judge LLMJudge, question, answer string) (float64, error) {
	prompt := fmt.Sprintf(`以下の「回答」が「質問」に対して適切に答えているか評価してください。

0（全く関係ない回答）〜 10（完璧に質問に答えている）の整数で回答してください。
数字のみを出力してください。

【質問】
%s

【回答】
%s`, question, answer)

	resp, err := judge.Judge(ctx, prompt)
	if err != nil {
		return 0, err
	}
	score, err := strconv.Atoi(strings.TrimSpace(resp))
	if err != nil {
		return 0, nil
	}
	return float64(score) / 10.0, nil
}

// ── Correctness: 回答が正解を含んでいるか ──

func Correctness(answer, groundTruth string) float64 {
	if strings.Contains(answer, groundTruth) {
		return 1.0
	}
	return 0.0
}

// ── レポート生成 ──

func Summarize(results []Result) Report {
	r := Report{Results: results}
	n := float64(len(results))
	if n == 0 {
		return r
	}
	for _, res := range results {
		if res.HitAtK {
			r.AvgHitRate += 1.0
		}
		r.AvgFaithful += res.Faithfulness
		r.AvgRelevancy += res.AnswerRelevancy
		r.AvgCorrect += res.Correctness
	}
	r.AvgHitRate /= n
	r.AvgFaithful /= n
	r.AvgRelevancy /= n
	r.AvgCorrect /= n
	return r
}
