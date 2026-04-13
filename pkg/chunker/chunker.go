// Package chunker はテキストを様々な戦略でチャンクに分割する。
//
// RAGの精度に最も直接的に影響する前処理。
// 分割戦略の選択は「正解」がなく、データの性質に合わせて実験で決める。
package chunker

import (
	"strings"
	"unicode/utf8"
)

// Chunk は分割されたテキスト断片。
type Chunk struct {
	Title    string // メタデータ（セクション名等）
	Content  string // テキスト本文
	Strategy string // どの戦略で生成されたか
	Index    int    // 元文書中の何番目のチャンクか
}

// ── 戦略1: 構造ベース分割 ──────────────────────────
// Markdownの「## 」見出しで分割する。
// 著者が意図した意味的な境界を尊重する。
//
// メリット: 意味的に一貫したチャンクが作れる
// デメリット: 見出しがない文書には使えない。セクションが大きすぎることがある
func ByHeading(doc string) []Chunk {
	var chunks []Chunk
	var title, content string
	idx := 0

	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "## ") {
			if strings.TrimSpace(content) != "" {
				chunks = append(chunks, Chunk{
					Title:    title,
					Content:  strings.TrimSpace(content),
					Strategy: "heading",
					Index:    idx,
				})
				idx++
			}
			title = strings.TrimPrefix(line, "## ")
			content = ""
		} else {
			content += line + "\n"
		}
	}
	if strings.TrimSpace(content) != "" {
		chunks = append(chunks, Chunk{
			Title:    title,
			Content:  strings.TrimSpace(content),
			Strategy: "heading",
			Index:    idx,
		})
	}
	return chunks
}

// ── 戦略2: 固定サイズ分割 ──────────────────────────
// 一定の文字数で機械的に切る。overlapで切れ目の情報欠落を防ぐ。
//
// メリット: どんな文書にも使える。実装が簡単
// デメリット: 文の途中でぶった切られる可能性がある
//
// size:    チャンクの文字数
// overlap: 前のチャンクと重複させる文字数
func ByFixedSize(doc string, size, overlap int) []Chunk {
	// 空白行・見出し記号を除去して本文だけにする
	var lines []string
	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		lines = append(lines, trimmed)
	}
	text := strings.Join(lines, "\n")

	var chunks []Chunk
	runes := []rune(text)
	total := len(runes)
	step := size - overlap
	if step <= 0 {
		step = size
	}

	for start := 0; start < total; start += step {
		end := start + size
		if end > total {
			end = total
		}
		content := string(runes[start:end])
		if strings.TrimSpace(content) == "" {
			continue
		}
		chunks = append(chunks, Chunk{
			Title:    "",
			Content:  strings.TrimSpace(content),
			Strategy: "fixed",
			Index:    len(chunks),
		})
		if end == total {
			break
		}
	}
	return chunks
}

// ── 戦略3: 段落ベース分割 ──────────────────────────
// 「### 」小見出し（条文単位）で分割する。
// 構造ベースの細粒度版。1つの条文が1チャンクになる。
//
// メリット: 1条文=1チャンクで最もピンポイントに検索できる
// デメリット: チャンクが小さすぎると文脈が失われる
func BySubHeading(doc string) []Chunk {
	var chunks []Chunk
	var parentTitle, title, content string
	idx := 0

	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "## ") {
			// 前の小セクションを保存
			if strings.TrimSpace(content) != "" {
				chunks = append(chunks, Chunk{
					Title:    parentTitle + " > " + title,
					Content:  strings.TrimSpace(content),
					Strategy: "subheading",
					Index:    idx,
				})
				idx++
			}
			parentTitle = strings.TrimPrefix(line, "## ")
			title = ""
			content = ""
		} else if strings.HasPrefix(line, "### ") {
			// 前の小セクションを保存
			if strings.TrimSpace(content) != "" {
				chunks = append(chunks, Chunk{
					Title:    parentTitle + " > " + title,
					Content:  strings.TrimSpace(content),
					Strategy: "subheading",
					Index:    idx,
				})
				idx++
			}
			title = strings.TrimPrefix(line, "### ")
			content = ""
		} else {
			content += line + "\n"
		}
	}
	if strings.TrimSpace(content) != "" {
		chunks = append(chunks, Chunk{
			Title:    parentTitle + " > " + title,
			Content:  strings.TrimSpace(content),
			Strategy: "subheading",
			Index:    idx,
		})
	}
	return chunks
}

// Stats はチャンクの統計情報を返す。
func Stats(chunks []Chunk) (count int, avgLen float64, minLen int, maxLen int) {
	count = len(chunks)
	if count == 0 {
		return
	}
	total := 0
	minLen = utf8.RuneCountInString(chunks[0].Content)
	maxLen = 0
	for _, c := range chunks {
		l := utf8.RuneCountInString(c.Content)
		total += l
		if l < minLen {
			minLen = l
		}
		if l > maxLen {
			maxLen = l
		}
	}
	avgLen = float64(total) / float64(count)
	return
}
