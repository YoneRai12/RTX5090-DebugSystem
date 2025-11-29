# RTX5090-DebugSystem

このリポジトリには、学習ジョブを監視し自動修復ループを回すためのスクリプト `phoenix_cli_manager.py` が含まれています。Python 3.11 以上と標準ライブラリのみで動作し、Gemini CLI（推奨）または Gemini API への `curl` 呼び出しを通じて外部LLMを利用します。

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

Gemini CLI の非対話モード（`gemini -p`）はツール実行や書き込みができないため、安全に自動修復ラッパーと組み合わせて使うことができます。Gemini API を使う場合は `curl` が必要です。

## 無人運転（Unattended Operation）ガイド

Windows タスクスケジューラ等で無人運用する場合の必須設定です。

### 1. 必須環境変数
```powershell
# 修正対象のホワイトリスト（必須: 空だと何も修正されません）
$env:PHOENIX_ALLOWLIST = "train.py;train_wrapper.py"

# トレーニングコマンド（フルパス推奨）
$env:PHOENIX_TRAIN_CMD = "python train_wrapper.py"

# ログフォーマット（JSONL推奨）
$env:PHOENIX_LOG_FORMAT = "json"

# LLM選択（Session 0 で CLI が動かない場合は API を使用）
$env:PHOENIX_PRIMARY = "gemini_api"
$env:GEMINI_API_KEY = "your_api_key"
```

### 2. タスクスケジューラ設定
- **プログラム/スクリプト**: `python.exe` (フルパス推奨)
- **引数**: `phoenix_cli_manager.py`
- **開始オプション (Start in)**: **必須**。プロジェクトのルートディレクトリ（`C:\path\to\RTX5090-DebugSystem`）を指定してください。これが空だと `System32` で起動し、権限エラーで即死します。
- **セキュリティ**: 「ユーザーがログオンしているかどうかにかかわらず実行する」（Session 0 互換）

### 3. 安全機能
- **Lock File**: 多重起動を防止します。
- **Atomic Shadow Patching**: パッチは一時ファイルでテストされ、成功時のみ適用されます。
- **Test Protection**: `tests/` ディレクトリは読み取り専用です。
- **OOM Handling**: Out of Memory エラーは1回だけ修正を試み、再発時は停止します。
