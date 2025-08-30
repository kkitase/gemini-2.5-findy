# Google Colab ではじめる Gemini API、今話題の画像生成 AI モデル「Nano Banana」を試してみよう。

「テキストから、あっという間にすごいクオリティの画像が作れたら…」
「写真の一部だけを、まるで魔法のように自然に修正したい…」

そんな願いを叶える、Google の最新画像生成 AI モデル、通称「Nano Banana」こと `gemini-2.5-flash-image-preview` が登場しました。
![alt text](../assets/data/banana.png)
この記事では、その驚くべき機能と使い方を、ハンズオン形式で学びます。[「Google Colab」](https://colab.research.google.com/) を使って、サンプルコードを実行しながら進めるので、誰でも簡単に最新の画像生成 AI を体験できます。


以下のボタンから Notebook を開いて進めましょう。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/03-jp-nano-banana.ipynb)

以降の解説は、Google Colab で実際にコードを実行しながら進めることを想定していますが、コードと解説を読み進めるだけでも学習できます。

## 重要: 環境の準備
- [セットアップと認証](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb) のセクションを完了し、`GEMINI_API_KEY` の設定が済んでいることを確認してください。
- もしエラーが出たら、[Gemini in Google Colab](https://colab.research.google.com/github/kkitase/gemini-2.5-findy/blob/main/notebooks/00-jp-setup-and-authentication.ipynb#scrollTo=7d140654) を使い、コードの説明やデバッグをして解決を試みてください。

## 1. Nano Banana (`gemini-2.5-flash-image-preview`) とは？

`gemini-2.5-flash-image-preview` は、Google が開発した最先端の画像生成・編集モデルです。テキストで指示するだけで高品質な画像を生成するだけでなく、既存の画像とテキストを組み合わせて、特定の部分だけを編集したり、複数の画像を合成したりすることもできます。

**主な機能:**
- **テキストから画像生成**: 詳細な説明文から、リアルな写真やイラストなどを生成します。
- **画像編集**: 元の画像に対して「この犬に帽子をかぶせて」のように指示し、一部分だけを自然に編集します。
- **複数画像の合成**: 複数の画像の良い部分を組み合わせて、新しい画像を創り出します。
- **対話による修正**: 一度の指示で終わらず、「もっと明るくして」のように対話しながら画像の完成度を高めていけます。

## 2. テキストから画像を生成する

まずは基本のテキストからの画像生成を試してみましょう。プロンプト（指示文）を工夫することで、驚くほど多彩な表現が可能です。

```python
# 必要なライブラリをインポート
from google import genai
from google.colab import userdata
from PIL import Image
from io import BytesIO

# APIキーを設定してクライアントを初期化
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

# 画像生成モデルのID
MODEL_ID = "gemini-2.5-flash-image-preview"

# 画像生成のためのプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
太陽の光で深いシワが刻まれた、温かく物知りな笑顔の高齢の日本の陶芸家の写実的なクローズアップポートレート。
彼は焼きたての茶碗を注意深く調べている。場所は素朴で日当たりの良い工房。
窓から差し込む柔らかく金色の夕方の光が、粘土の細かい質感を際立たせている。
85mmのポートレートレンズで撮影され、背景は柔らかくぼけている（ボケ）。
全体の雰囲気は穏やかで達人のよう。縦長のポートレート。
"""

# 変数を初期化
image = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果から画像データを抽出し、表示・保存する
# 応答には複数のパート（テキストや画像など）が含まれている可能性があるため、ループで処理します
for part in response.candidates[0].content.parts:
  # パートが画像データの場合
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      # 画像データを読み込んで開きます
      image = Image.open(BytesIO(part.inline_data.data))
      # 保存するファイル名を定義します
      image_filename = 'generated_potter.png'
      # 画像をファイルとして保存します
      image.save(image_filename)
      print(f"生成された画像を '{image_filename}' として保存しました。")

# Colab上に画像を表示
display(image)
```

## 3. プロンプトのコツ

Nano Banana の能力を最大限に引き出すには、プロンプトの書き方が非常に重要です。単語を並べるだけでなく、**情景を物語のように説明する**のが基本です。

### 特定のスタイルで描いてほしい時

「ステッカースタイルで」「水彩画風で」のように、スタイルをはっきり指定します。背景を透明にしたい場合は「背景は白で」と伝えるのが効果的です。

```python
# 画像生成のためのプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
かわいいステッカーアート。幸せそうなレッサーパンダが小さな竹の帽子をかぶり、笹の葉を食べている。
太くてクリーンな線、セル画風のシンプルな陰影、鮮やかな色使い。
背景は白。
"""

# 変数を初期化
image = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果から画像データを抽出し、表示・保存する
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      image = Image.open(BytesIO(part.inline_data.data))
      image_filename = 'generated_sticker.png'
      image.save(image_filename)
      print(f"生成された画像を '{image_filename}' として保存しました。")

# Colab上に画像を表示
display(image)
```

```python
# TODO
# 水彩画風や、油絵風など、他のスタイルも試してみましょう。
```

## 4. 画像を編集する

Nano Banana のもう一つの強力な機能が、既存の画像と言葉の指示を組み合わせて編集する機能です。

まず、編集したい画像を Colab にアップロードしておきましょう。

```python
# 編集したい画像を準備（サンプルとして猫の画像をダウンロード）
!curl -o my_cat.jpg "https://storage.googleapis.com/generativeai-downloads/images/cat.jpg"

# 画像を開いて表示
image_to_edit = Image.open('my_cat.jpg')
display(image_to_edit)
```

次に、その画像に対して「何をしてほしいか」をテキストで具体的に指示します。

```python
# 画像編集のプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = "この猫に、パーティー用の小さな帽子をかぶせてください。背景や猫の他の部分は変えないでください。"

# 変数を初期化
edited_image = None

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, image_to_edit], # テキストプロンプトと画像オブジェクトをリストで渡す
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      edited_image = Image.open(BytesIO(part.inline_data.data))
      edited_image.save('edited_cat.png')
      print("編集された画像を 'edited_cat.png' として保存しました。")

# Colab上に画像を表示
display(edited_image)
```


```python
# TODO
# 他にも「この猫にサングラスをかけて」「この風景の空を夕焼けに変えて」など、様々な編集を試してみましょう。
```

このように、変更したい部分と、変更したくない部分を明確に指示することで、狙い通りの編集が可能になります。

> **注**: 生成されたすべての画像には、信頼性検証のために SynthID ウォーターマークが含まれています。詳細は[公式ドキュメント](https://ai.google.dev/gemini-api/docs/image-generation?hl=ja) をご覧ください。

## 5. Webサイトのランディングページをデザインする

Nano Banana は、具体的なUIコンポーネントやレイアウト構造も理解できます。これを利用して、Webサイトのランディングページ（LP）のデザイン案を生成させてみましょう。

## 5. WebサイトのUIデザイン

Nano Banana は、具体的な UI コンポーネントやレイアウト構造も理解できます。これを利用して、Web サイトのランディングページ（LP）のデザイン案を生成させてみましょう。

### 5.1. 基本的なLPデザイン

まずは、基本的な旅行アプリのLPをデザインします。

```python
# Webサイトデザインのプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
モダンでミニマルな旅行アプリのランディングページのUIデザイン。
ヒーローセクションには「Explore the World Unknown」というキャッチコピーと、夕暮れの美しいビーチの背景画像。
その下には「Search by Destination」「Search by Theme」「Search by Budget」という3つの特徴的な検索カードを配置。
全体の配色は青と白を基調とし、クリーンで信頼感のある印象を与える。
"""

# 変数を初期化
web_design_image = None # Initialize the variable

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      web_design_image = Image.open(BytesIO(part.inline_data.data))
      web_design_image.save('generated_lp_design.png')
      print("生成されたWebデザイン画像を 'generated_lp_design.png' として保存しました。")
      break # Assuming only one image is expected

# Google Colab上に画像を表示
display(web_design_image)
```

### 5.2. メニューの追加

次に、生成したLPにグローバルナビゲーションメニューを追加してみましょう。

```python
# メニュー追加のプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
モダンでミニマルな旅行アプリのランディングページのUIデザイン。
ヘッダーに「Home」「Destinations」「About Us」「Contact」のナビゲーションメニューを追加。
ヒーローセクションには「Explore the World Unknown」というキャッチコピーと、夕暮れの美しいビーチの背景画像。
その下には「Search by Destination」「Search by Theme」「Search by Budget」という3つの特徴的な検索カードを配置。
全体の配色は青と白を基調とし、クリーンで信頼感のある印象を与える。
"""

# 初期化
web_design_image_with_menu = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      web_design_image_with_menu = Image.open(BytesIO(part.inline_data.data))
      web_design_image_with_menu.save('generated_lp_design_with_menu.png')
      print("メニュー付きのWebデザイン画像を 'generated_lp_design_with_menu.png' として保存しました。")
      break

# Google Colab上に画像を表示
display(web_design_image_with_menu)
```

### 5.3. バナーの変更

最後に、ヒーローセクションの背景画像を、より具体的なテーマに変更してみましょう。

```python
# バナー変更のプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
モダンでミニマルな旅行アプリのランディングページのUIデザイン。
ヘッダーに「Home」「Destinations」「About Us」「Contact」のナビゲーションメニュー。
ヒーローセクションには「Explore the World Unknown」というキャッチコピーと、日本の桜並木の美しい背景画像。
その下には「Search by Destination」「Search by Theme」「Search by Budget」という3つの特徴的な検索カードを配置。
全体の配色はピンクと白を基調とし、春らしく華やかな印象を与える。
"""

# 変数を初期化
web_design_image_cherry_blossom = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      web_design_image_cherry_blossom = Image.open(BytesIO(part.inline_data.data))
      web_design_image_cherry_blossom.save('generated_lp_design_cherry_blossom.png')
      print("桜のWebデザイン画像を 'generated_lp_design_cherry_blossom.png' として保存しました。")
      break

# Google Colab上に画像を表示
display(web_design_image_cherry_blossom)
```

## 6. 広告用の商品写真を生成し、対話的に編集する

新商品の広告キャンペーンで使うような、プロフェッショナルで魅力的な画像を生成し、さらにそこから対話を通じて修正を加えていく過程を体験してみましょう。

まずは、ベースとなる飲料の広告写真を作成します。

```python
# 広告写真のプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
製品広告写真：新しいエナジードリンク「SPARK」のアルミ缶。
缶は冷たく、たくさんの水滴がついており、砕いた氷の上に置かれている。
背景はシャープでモダンな青いグラデーション。
缶には「SPARK」の文字がスタイリッシュなフォントで描かれていること。
エネルギッシュで爽快なイメージで、非常に高画質なこと。
"""

# 変数を初期化
ad_image = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image = Image.open(BytesIO(part.inline_data.data))
      ad_image.save('generated_drink_ad.png')
      print("生成された広告画像を 'generated_drink_ad.png' として保存しました。")

# Colab上に画像を表示
display(ad_image)
```

### 6.1. 生成した画像に人物を追加する

次に、この商品写真に人物を追加して、より魅力的な広告にしてみましょう。「この缶を人が持っているように」と指示します。

```python
# 人物を追加するプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
製品広告写真：このドリンクの缶を、スポーティーな若い女性が笑顔で持っているように編集してください。
商品の見え方や背景の雰囲気は維持してください。
"""

# 変数を初期化
ad_image_with_person = None

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image], # テキストプロンプトと元の画像を渡す
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_with_person = Image.open(BytesIO(part.inline_data.data))
      ad_image_with_person.save('ad_with_person.png')
      print("人物を追加した画像を 'ad_with_person.png' として保存しました。")

# Colab上に画像を表示
display(ad_image_with_person)
```

### 6.2. 商品のロゴを変更する

急なデザイン変更にも対応できます。缶に描かれたロゴを、別の名前に変更してみましょう。

```python
# ロゴを変更するプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
製品広告写真：缶に書かれている「SPARK」というロゴを、新しい「AQUA」というロゴに自然な形で変更してください。
フォントのスタイルは元のロゴに似せてください。
"""

# 変数を初期化
ad_image_new_logo = None

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image_with_person], # 人物追加後の画像をベースにする
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_new_logo = Image.open(BytesIO(part.inline_data.data))
      ad_image_new_logo.save('ad_new_logo.png')
      print("ロゴを変更した画像を 'ad_new_logo.png' として保存しました。")

# Colab上に画像を表示
display(ad_image_new_logo)
```

### 6.3. 背景の色を調整する

最後に、商品のフレーバーに合わせて背景色を変更してみましょう。オレンジ味を想定して、背景を暖色系に変えてみます。

```python
# 背景色を変更するプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
背景全体を、暖色系のオレンジ色の鮮やかなグラデーションに変更してください。人物や商品には影響を与えないでください。
"""
# 上記プロンプトで写真を生成できない時は英語にしてみてください。

# 変数を初期化
ad_image_final = None

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image_new_logo], # ロゴ変更後の画像をベースにする
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_final = Image.open(BytesIO(part.inline_data.data))
      ad_image_final.save('ad_final.png')
      print("最終版の広告画像を 'ad_final.png' として保存しました。")

# Colab上に画像を表示
display(ad_image_final)
```
このように、一度画像を生成してから、対話を繰り返すことで、細かな要望を反映させながら完成度を高めていくことができます。

## 7. AIで生成した人物のポーズを、スケッチで変更する

まず、ベースとなる人物画像をAIで生成します。今回は「ヨガの木のポーズをしている男性」を生成してみましょう。

```python
# プロンプトを定義
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
以下の要件で画像を生成してください：

- 被写体: ヨガの木のポーズをしている男性の全身
- 場所: 静かなヨガスタジオ
- 背景: ミニマルで明るい
- スタイル: プロのカメラマンが高品質な機材で撮影したような、リアルな写真
"""

# 英語プロンプトの例
# prompt = """
# A full-body photograph of a man in the tree yoga pose.
# The setting is a quiet yoga studio with a minimal and bright background.
#The image should be realistic, as if taken by a professional photographer with high-quality equipment.
# """

# 変数を初期化
person_image = None

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果から画像を抽出し、後続の処理で使えるように変数に格納する
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      person_image = Image.open(BytesIO(part.inline_data.data))
      person_image.save('generated_yoga_person_male.png')
      print("生成された人物画像を 'generated_yoga_person_male.png' として保存しました。")
      break

# 生成された画像を表示
print("生成された元になる人物写真:")
display(person_image)
```

次に、この生成した男性の画像に、簡単な棒人間のスケッチで別のポーズをとらせてみましょう。

```python
# ポーズ指示に使うスケッチのパス
sketch_image_path = '/content/gemini-2.5-findy/assets/data/boningen4.jpeg'
sketch_image = Image.open(sketch_image_path)

print("指示に使うスケッチ:")
display(sketch_image)

# スケッチを使った画像編集のプロンプト
# 注意: 下記プロンプトで写真を生成できない時は英語にしてみてください。
prompt = """
ポーズを変更して
左の男性に、右のスケッチと同じポーズをとらせてください。
"""

# 変数を初期化
posed_image = None

# モデルを呼び出して画像を編集
# person_imageがNoneでないことを確認
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, person_image, sketch_image], # テキスト、スケッチ画像、生成した人物画像の3つをリストで渡す
)

# 結果を表示
for part in response.candidates[0].content.parts:
    if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
        posed_image = Image.open(BytesIO(part.inline_data.data))
        posed_image.save('posed_person_male.png')
        print("ポーズを編集した画像を 'posed_person_male.png' として保存しました。")
        break

display(posed_image)
```