# RTX5090-DebugSystem

このリポジトリには、学習ジョブを監視し自動修復ループを回すためのスクリプト `phoenix_cli_manager.py` が含まれています。Python 3.11 以上と標準ライブラリのみで動作し、Gemini CLI（推奨）または Gemini API への `curl` 呼び出しを通じて外部LLMを利用します。

- stdout/stderr のハートビート監視で無出力ハングを検出し、プロセスグループを kill
- LLM 応答は `unified_diff`/`replace_range` パッチ配列限定で安全性を向上
- バックアップ/ログの上限サイズ管理と原子的な書き込み、`py_compile`＋任意テストコマンドで検証
- `.phoenix_cli/train.lock` による排他ロックで GPU/作業ディレクトリを 1 ジョブに限定し、ハングによる終了理由もエラーシグネチャへ組み込む

## 使い方
1. 必要に応じて環境変数を設定し、学習コマンドを `Config.train_cmd`（デフォルト: `python train.py`）で指定します。
2. Gemini CLI を使う場合は `gemini` コマンドにログイン済みの状態で `PHOENIX_PRIMARY=gemini_cli` を指定します。Gemini API を使う場合は `GEMINI_API_KEY` を設定し `PHOENIX_PRIMARY=gemini_api` を指定します。
3. `python phoenix_cli_manager.py` を実行すると、学習コマンドを起動し、異常終了時に自動で修正・再実行を試みます。ログは `.phoenix_cli/run.log` に出力され、バックアップは `.phoenix_cli/backups/` に保存されます。標準出力／標準エラーの両方が一定時間（デフォルト 15 分）止まった場合はハングと判定してプロセスツリーを強制終了し、自動修復ループに入ります。

## 主な環境変数
- `PHOENIX_PRIMARY`: 使用するLLMクライアント（`gemini_cli` または `gemini_api`）。
- `PHOENIX_DRY_RUN`: `1` の場合、ファイルは上書きせずバックアップのみ作成します。
- `GEMINI_API_KEY`: Gemini API を利用する際の API キー（名前は `GEMINI_API_KEY_ENV` で変更可能）。
- `GEMINI_MODEL`: Gemini API のモデル名（デフォルト: `gemini-2.0-flash`）。
- `PHOENIX_MAX_RETRIES`: 同一エラーシグネチャに対する最大修正試行回数。
- `PHOENIX_TAIL`: LLM へ渡すログの末尾行数。
- `PHOENIX_MAX_FILE_CHARS` / `PHOENIX_MAX_PROMPT_CHARS`: プロンプトに含めるソースやプロンプトの最大文字数。
- `PHOENIX_HEARTBEAT_MIN`: 標準出力／標準エラーの無出力が何分続いたらハングと判定するか（0 で無効）。
- `PHOENIX_TEST_CMD`: 修正適用後に実行する最小テストコマンド（例: `pytest -q` など）。
- `PHOENIX_ALLOWLIST`: 修正を許可するファイルパターン（セミコロン区切り）。
- `PHOENIX_LOG_MAX_BYTES`: ローテーション前のログ最大バイト数（デフォルト 5MB）。
- `PHOENIX_MAX_BACKUPS`: バックアップ世代数の上限。
- `PHOENIX_COOLDOWN_SECONDS`: 再修復不能で停止する前に待機する秒数。
- `PHOENIX_LOCK_STALE`: ロックが残っている場合に、PID が生存しておらず一定秒数以上経過していれば破棄を許可する猶予秒（0 で無効）。

Gemini CLI の非対話モード（`gemini -p`）はツール実行や書き込みができないため、安全に自動修復ラッパーと組み合わせて使うことができます。Gemini API を使う場合は `curl` が必要です。

安全運用上のヒント:
- 自動復旧ランナーとリポジトリの自動更新タスク（例: ORA 更新）を同じ権限・同じワークツリーで動かさない。更新は別ユーザー/別環境で検証し、`train.lock` で学習ジョブとの競合を避ける
- 公開リポジトリに対する自己ホストランナー的な運用は避け、未知の PR から任意コードが実行されない構成にする
- ハードウェア監視（温度・電力・ディスク残量など）と併用し、閾値超過時は学習を止める
