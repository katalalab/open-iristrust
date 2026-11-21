# open-iristrust Agent Rules

共通ルールは各AIのユーザー規則（`~/.claude/CLAUDE.md` / `~/.codex/AGENTS.md`）に従う。ここにはこのリポジトリ固有の事項だけを書く。

- 役割: Worldcoin Open IRIS のアルゴリズムと公開モデルをもとにした、Python を使わない Rust 製の虹彩認証ライブラリと CLI。元実装の取り込み範囲は `THIRD_PARTY_NOTICES.md` に従う。
- 検証（README と同じ。CI は `.github/workflows/ci.yml` にあり、ここでは触れない）:
  - `cargo fmt --all -- --check`
  - `cargo clippy --locked --all-targets --all-features -- -D warnings`
  - `cargo test --locked` と `cargo test --locked --no-default-features`
  - ONNX モデルが要るテストは `cargo test --locked --test onnx -- --ignored --nocapture`（モデルは `scripts/fetch-model.sh` で取得する明示的な手順）。
- 抽出・照合の処理から外部ネットワークへ接続しない。モデルの取得だけが別の明示的な手順であり、この性質を変えない。
- 実際の虹彩画像・テンプレートなど生体データをコミットしない。例は合成データ（`demo`）だけを使う。
- ブランチは 1 機能 1 枝 1 PR。main へ直接 push しない。merge は人間が行う。
