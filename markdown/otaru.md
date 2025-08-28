# Google Colab ではじめる Gemini、AI ツール活用からエージェント開発まで。

「Google の AI、Gemini を使ってみたいけど、何から始めればいいかわからない…」
「簡単なツールは使ったことがあるけど、もっと本格的な開発にも挑戦してみたい！」

そんな風に思っていませんか？このコースでは、プログラミング不要のツールで AI に触れる楽しさから、簡単なプログラミング、そして自律的にタスクを実行する「AI エージェント」の開発まで、つまずくことなくステップアップできるカリキュラムを提供します。

AI 活用の具体的なイメージを掴み、「AI に触れる楽しさ」から「AI を創る喜び」へ、一緒に歩みを進めましょう。

以下のボタンから Notebook を開いて進めましょう。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/xx-jp-gemini-master-course.ipynb)

以降の解説は、Google Colab で実際にコードを実行しながら進めることを想定していますが、コードと解説を読み進めるだけでも学習できます。

## 学習目標

* Gemini のツール群（NotebookLM, Gemini CLI など）を使いこなし、情報収集や資料作成を効率化できる。
* Gemini API の基本的な使い方を理解し、Google Colab 上で簡単な AI アプリケーションを開発できる。
* AI エージェントの基本概念を理解し、Function Calling などを用いて特定のタスクを自動化するプログラムを作成できる。

---

## 1. Gemini ファミリー探求

まずは、プログラミングなしで使えるツールを通して、Gemini の能力を体感し、AI を活用する感覚を掴みましょう。このセクションでは、コーディング不要のツールに触れ、AI 活用の勘所を掴むことを目指します。

### 1.1. 対話 AI の基本「Gemini」

すべての基本となる対話 AI「Gemini」とのコミュニケーション方法を学びます。効果的な指示（プロンプト）の書き方をマスターすれば、AI はあなたの最高のパートナーになります。

```python
# TODO:
# Gemini に自己紹介をさせてみましょう。
# 「あなたは誰ですか？何ができますか？」と聞いてみましょう。
```

### 1.2. 情報アシスタント「NotebookLM」

PDF や Web サイトの URL などの資料を読み込ませるだけで、あなた専用の専門家になってくれる「NotebookLM」。RAG (Retrieval-Augmented Generation) という技術を手軽に体験できます。

*   **活用例:**
    *   長文の議事録を読み込ませて、要点をまとめてもらう。
    *   複数の論文を読み込ませて、関連研究をリサーチしてもらう。

### 1.3. 自律型リサーチツール「Deep Research」(仮称)

一つの問いを投げかけるだけで、AI が自ら必要な情報を収集し、分析し、レポートを作成してくれます。AI が自律的に思考し、行動する「エージェント」の概念に触れてみましょう。

### 1.4. CUI で AI を操る「Gemini CLI」

エンジニアにとって最も身近な CUI（コマンドライン・インターフェース）で Gemini を操作します。ターミナル上での対話や、日々のコマンドライン作業との連携方法を探ります。

---

## 2. コーディングの第一歩、Colab で AI アプリ開発

いよいよプログラミングの世界へ。このセクションでは、Gemini API の基本を学び、自分で簡単な AI アプリを作る成功体験を得ることを目指します。Google Colab を使って、Gemini API を呼び出し、自分だけの AI アプリケーションを作成します。

### 2.1. 開発環境の準備

まずは、開発の準備を整えます。Google Colab の使い方に慣れ、Gemini API を使うための「鍵」となる API キーを取得します。

*   [00-jp-setup-and-authentication.md](https://github.com/kkitase/gemini-2.5-findy/blob/main/markdown/00-jp-setup-and-authentication.md) を参考に環境を準備してください。

### 2.2. Gemini API を使ってみよう

API を使って、テキストを生成したり、画像を認識させたりする方法を学びます。

```python
# Gemini API を Python で利用するためのライブラリをインポート
from google import genai
from google.colab import userdata

# API キーを使ってクライアントを作成
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

# テキスト生成の例
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="AI アプリ開発のアイデアを5つ教えて！"
)
print(response.text)
```

### 2.3. ハンズオン：初めての AI アプリ開発

学んだ知識を活かして、簡単な AI アプリケーションを開発してみましょう。

**演習1：ブログ記事のタイトル提案アプリ**
```python
# TODO:
# 「Gemini API の活用法」というテーマで、魅力的なブログ記事のタイトルを10個提案させるプロンプトを作成してください。
prompt = "..."

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
print(response.text)
```

**演習2：画像説明アプリ**
```python
# TODO:
# 好きな画像をアップロードし、その画像が何であるかを説明させるコードを書いてみましょう。
# ヒント: 02-jp-multimodal-capabilities.md の「1. 画像理解: 単一の画像」が参考になります。
```

---

## 3. AI エージェント開発入門

AI に「道具」を与え、自律的にタスクを解決させる「AI エージェント」の開発に挑戦します。このセクションでは、AI が外部ツールを使いこなし、自律的にタスクをこなす「エージェント」の仕組みを学びます。

### 3.1. AI エージェントとは？

AI が「思考 → 行動 → 観察」のループを繰り返しながら、目標達成に向けて自律的に動作する仕組みを学びます。

### 3.2. AI に道具を与える「Function Calling」

AI が Python の関数などの外部ツールを呼び出すための「Function Calling」という重要な技術を学びます。これにより、AI は単なる対話相手から、実際に作業を実行するパートナーへと進化します。

### 3.3. ハンズオン：初めての AI エージェント開発

**演習1：お天気アドバイザーエージェント**
```python
# TODO:
# 現在地の天気を取得する架空の関数 `get_weather(location)` を定義し、
# Function Calling を使って Gemini にその関数を呼び出させ、
# 天気に合わせた服装を提案させるエージェントを作成してみましょう。
```

**演習2：リサーチエージェント**
```python
# TODO:
# Google 検索を実行する架空の関数 `search_google(query)` を定義し、
# Function Calling を使って Gemini に「今日の AI 業界のトップニュース」を検索させ、
# 結果を要約させるエージェントを作成してみましょう。
```

---

## 4. さらなる高みへ

このコースで得た知識は、広大な AI 開発の世界への入り口です。ここでの学びを土台に、次のステップへ進むための道筋を示します。

*   **応用トピック:**
    *   **Vertex AI:** エンタープライズ向けの本格的な AI 開発プラットフォーム
    *   **高度な RAG:** より精度の高い情報検索・回答生成技術
    *   **自律型エージェント開発:** LangChain や LlamaIndex などのフレームワークを活用した高度なエージェント構築

*   **最終課題:**
    *   「あなたの日常業務や学習を自動化する、オリジナルの AI エージェント」を企画し、開発してみましょう！

---