# RTX5090-DebugSystem ユーザーガイド

このシステムは、**「寝ている間にAIが勝手にバグを直して学習を続けてくれる」** 自動運転ツールです。
特に RTX5090 などの強力なGPUマシンを、**自宅**に置いて無人で回し続けるために設計されています。

## 🛠️ 仕組み（どうやって動くの？）

このシステムは `phoenix_cli_manager.py` というマネージャーが司令塔になります。

1.  **監視 (Monitor)**:
    *   あなたの学習スクリプト（例: `train.py`）を起動し、じっと見守ります。
    *   ログが出なくなったり（ハングアップ）、エラーで落ちたり（クラッシュ）するのを検知します。

2.  **診断 (Analyze)**:
    *   エラーが起きると、そのエラーログとソースコードを LLM (Gemini) に送信します。
    *   「このエラーはどう直せばいい？」と相談します。

3.  **手術 (Shadow Patching)**:
    *   **ここが安全ポイント！** いきなり本番ファイルを書き換えません。
    *   まず「影武者（コピーしたファイル）」に修正を適用します。
    *   その影武者でテストを実行し、本当に直ったか、他を壊していないか確認します。

4.  **適用 & 再開 (Commit & Restart)**:
    *   テストに合格したら、本番ファイルを書き換えます（アトミック更新）。
    *   そして学習を再開します。

これを繰り返すことで、夜中にエラーで止まっても、朝には直って学習が進んでいる状態を目指します。

---

## 🧠 最適化戦略（2つのループ）

このシステムは、Discord Botとして対話しながら自分を賢くするために、2つのループを回します。

1.  **オンライン最適化（即効性）**:
    *   Discord チャット中は**重み更新しません**。
    *   代わりに、あなたの好みや直近の会話を**メモリ**から検索し、プロンプトに注入することで即座に振る舞いを修正します。

2.  **オフライン最適化（じわじわ賢く）**:
    *   重み更新（LoRA学習）は、**夜間にまとめて**行います。
    *   `phoenix_cli_manager.py` が学習プロセスを監視し、エラーが出ても自動修復しながら学習を完遂させます。
    *   更新されたモデルは、テストに合格した時のみ反映されます。

---

## 🚀 使い方（3ステップ）

### ステップ 1: 準備

まず、修正してほしいファイルと、実行コマンドを環境変数で教えます。
PowerShell での設定例です。

```powershell
# 1. 修正を許可するファイル（これ以外は絶対触らせない安全装置）
$env:PHOENIX_ALLOWLIST = "train.py;models/*.py"

# 2. 実行する学習コマンド
$env:PHOENIX_TRAIN_CMD = "python train.py --batch_size 32"

# 3. Gemini の設定 (APIキーがある場合)
$env:PHOENIX_PRIMARY = "gemini_api"
$env:GEMINI_API_KEY = "あなたのAPIキー"
```

### ステップ 2: 起動

マネージャー経由で学習をスタートします。

```powershell
python phoenix_cli_manager.py
```

これだけです！
画面には学習ログが表示され、エラーが起きると自動で「修復モード」に入ります。

### 9. 📊 高度なレポート機能 (Advanced Reporting)
学習の進捗やLossの推移を可視化し、Discordで確認できます。

**機能:**
*   **グラフ自動生成**: `matplotlib` がインストールされている場合、Lossの推移グラフを自動生成します。
*   **!report コマンド**: Discordから `!report` と入力すると、現在の進捗とLossグラフ（画像）を返信します。
*   **Fixer Model Logging**: どのモデル（Gemini Pro / Local LLM）が修正に成功したかをログに記録し、分析に役立てます。

### 10. 🛡️ フォールバック保護 (Fallback Guard)
ローカルLLMへの過剰な負荷を防ぐため、クールダウン機能を搭載しています。

```powershell
# フォールバック実行後のクールダウン時間 (分)
$env:PHOENIX_FALLBACK_COOLDOWN_MIN = "10"
```

## 推奨環境
*   **GPU**: NVIDIA RTX 3090 / 4090 / 5090 (VRAM 24GB+)
*   **OS**: Windows 10/11 (WSL2推奨) or Linux
*   **Python**: 3.10+
*   **推奨ライブラリ**: `matplotlib` (グラフ生成用)

### ステップ 3: 無人運転（タスクスケジューラ）

**自宅**で放置する場合は、Windowsの「タスクスケジューラ」を使います。

*   **プログラム**: `python.exe` (フルパス推奨)
*   **引数**: `phoenix_cli_manager.py`
*   **開始オプション (Start in)**: **重要！** このフォルダのパス（`C:\...\RTX5090-DebugSystem`）を必ず入れてください。
*   **セキュリティ設定**: 「ユーザーがログオンしているかどうかにかかわらず実行する」にチェック。

これで、PCが再起動しても勝手に裏で学習が回ります。

---

## 🛡️ 安全機能（なぜ暴走しないの？）

AIにコードを書き換えさせるのは怖いですよね？ だからガチガチの安全装置を付けました。

1.  **テスト保護**: `tests/` フォルダは読み取り専用です。AIがテストを改ざんして「合格しました！」と嘘をつくのを防ぎます。
2.  **変更制限**: 1回に変更できるのは 50行、3ファイルまで。大規模な破壊を防ぎます。
3.  **ロックファイル**: 間違って2回起動しても、2つ目はすぐに終了します。GPUの取り合いになりません。
4.  **OOM対策**: 「メモリ不足 (Out of Memory)」の場合は、1回だけ「バッチサイズ下げてみる？」等の修正を試しますが、それでもダメなら諦めて停止します。無限ループでGPUを痛めるのを防ぎます。

## 📂 ログの見方

*   `.phoenix_cli/run.log`: 全体の動作ログ（JSON形式）。いつエラーが起きて、どう直したかが記録されます。
*   `.phoenix_cli/backups/`: 書き換えられる前のファイルのバックアップがここに残ります。いつでも元に戻せます。

## 🆘 フォールバック機能 (Fallback)

Gemini や Grok などの外部APIがダウンした場合や、ネットワークが切断された場合に、**RTX5090 上のローカルLLM** に自動で切り替えて修正を継続できます。

### 設定方法

#### 1. CLIモード (コマンド実行)
環境変数 `PHOENIX_FALLBACK_LLM_CMD` に、ローカルLLMを呼び出すコマンドを設定します。
**セミコロン区切りで複数指定可能**です。1つ目が失敗したら2つ目を試します。

```powershell
# 例: まず StarCoder を試し、ダメなら CodeLlama を試す
$env:PHOENIX_FALLBACK_LLM_CMD = "python run_starcoder.py; python run_codellama.py"
```

#### 2. APIモード (LM Studio / vLLM / llama.cpp)
OpenAI互換のAPIサーバーがローカルで動いている場合（例: LM Studio）、以下のように設定します。
モデル名はVRAM容量に応じて**自動設定**されますが、手動で指定も可能です。

```powershell
$env:PHOENIX_FALLBACK_LLM_TYPE = "api"
$env:PHOENIX_FALLBACK_LLM_URL = "http://localhost:1234/v1/chat/completions"
# $env:PHOENIX_FALLBACK_LLM_MODEL = "local-model" # 省略すると自動判定
```

# ドライラン (ファイル書き換えを行わず、ログ出力のみ)
$env:PHOENIX_DRY_RUN = "1"; python phoenix_cli_manager.py
```

ログは `.phoenix_cli/run.log` に出力され、変更されたファイルのバックアップは `.phoenix_cli/backups/` に保存されます。

## 環境変数設定

動作に必要な主要な環境変数は以下の通りです。

| 変数名 | 説明 | デフォルト値 |
| :--- | :--- | :--- |
| `PHOENIX_TRAIN_CMD` | 実行する学習コマンド | `python train.py` |
| `PHOENIX_PRIMARY` | 使用するLLMクライアント (`gemini_cli` / `gemini_api`) | `gemini_cli` |
| `GEMINI_API_KEY` | Gemini API キー | - |
| `PHOENIX_ALLOWLIST` | 修正を許可するファイルパターン (glob) | `*.py` |
| `PHOENIX_MAX_RETRIES` | 同一エラーに対する最大修正試行回数 | `3` |
| `PHOENIX_HEARTBEAT_MIN` | 無出力状態が続いた場合のタイムアウト (分) | `15` |

## 高度な機能設定

### 1. Discord 連携 (通知・操作)
Webhookによる通知およびBotによるリモート操作が可能です。

```powershell
$env:PHOENIX_DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
$env:PHOENIX_DISCORD_BOT_TOKEN = "..."      # Bot Token (操作用)
$env:PHOENIX_DISCORD_CHANNEL_ID = "..."     # Channel ID (操作用)
```
**対応コマンド**: `!status` (状況確認), `!stop` (停止), `!resume` (再開), `!config` (設定確認)

### 2. 進捗・品質監視
学習ログを解析し、進捗状況やLossの異常を検知します。

```powershell
$env:PHOENIX_NOTIFY_STEPS = "25"            # 25ステップ毎に通知
$env:PHOENIX_LOSS_REGEX = "loss[:=]\s*([\d\.]+)" # Loss抽出用正規表現
```

### 3. 安全運用・デバッグ (Patch Mode)
本番投入前の動作確認用モードです。

```powershell
$env:PHOENIX_PATCH_MODE = "auto"            # auto / restart-only / analyze-only
$env:PHOENIX_FALLBACK_MAX_PER_RUN = "5"     # フォールバック実行回数上限
```

### 4. データ管理 (Data Verification)
`data/` ディレクトリ内のデータ整合性を起動時に検証します。

```powershell
$env:PHOENIX_DATA_DIR = "data"
```

## トラブルシューティング

**システムが不安定な場合 (BSOD等)**
高負荷な学習処理と監視プロセスの競合、あるいはハードウェアの電力不足が疑われます。

1.  **ドライバ**: GPUドライバを最新版に更新してください。
2.  **電源**: ピーク電力に対応した電源ユニットであることを確認してください。
3.  **監視無効化**: `PHOENIX_MAX_GPU_TEMP=100` を設定し、温度監視 (`nvidia-smi` 呼び出し) を無効化して切り分けを行ってください。

## ライセンス
GNU General Public License v3.0 (GPL-3.0)
