# Google Colab ではじめる Gemini API、今話題の画像生成 AI モデル「Nano Banana」を試してみよう。

「テキストから、あっという間にすごいクオリティの画像が作れたら…」
「写真の一部だけを、まるで魔法のように自然に修正したい…」

そんな願いを叶える、Google の最新画像生成 AI モデル、通称「Nano Banana」こと `gemini-2.5-flash-image-preview` が登場しました。この記事では、その驚くべき機能と使い方を、ハンズオン形式で学びます。[「Google Colab」](https://colab.research.google.com/) を使って、サンプルコードを実行しながら進めるので、誰でも簡単に最新の画像生成 AI を体験できます。

さあ、あなたも言葉を絵に変える魔法を、その手で体験してみませんか？

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

# 画像生成のためのプロンプト（指示文）
prompt = """
太陽の光で深いシワが刻まれた、温かく物知りな笑顔の高齢の日本の陶芸家の写実的なクローズアップポートレート。
彼は焼きたての茶碗を注意深く調べている。場所は素朴で日当たりの良い工房。
窓から差し込む柔らかく金色の夕方の光が、粘土の細かい質感を際立たせている。
85mmのポートレートレンズで撮影され、背景は柔らかくぼけている（ボケ）。
全体の雰囲気は穏やかで達人のよう。縦長のポートレート。
"""

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
# 画像生成のためのプロンプト（指示文）
prompt = """
小さな竹の帽子をかぶった、幸せそうなレッサーパンダのかわいいスタイルのステッカー。
緑の笹の葉をもぐもぐ食べている。
デザインは太くクリーンな輪郭線、シンプルなセル画風の陰影、そして鮮やかなカラーパレットが特徴。
背景は必ず白にすること。
"""

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
prompt = "この猫に、パーティー用の小さな帽子をかぶせてください。背景や猫の他の部分は変えないでください。"

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

このように、変更したい部分と、変更したくない部分を明確に指示することで、狙い通りの編集が可能になります。

> **注**: 生成されたすべての画像には、信頼性検証のために SynthID ウォーターマークが含まれています。詳細は[公式ドキュメント](https://ai.google.dev/gemini-api/docs/image-generation?hl=ja) をご覧ください。

## 5. Webサイトのランディングページをデザインする

Nano Banana は、具体的なUIコンポーネントやレイアウト構造も理解できます。これを利用して、Webサイトのランディングページ（LP）のデザイン案を生成させてみましょう。

```python
# Webサイトデザインのプロンプト
prompt = """
モダンでミニマルな旅行アプリのランディングページのUIデザイン。
ヒーローセクションには「あなたの知らない世界へ」というキャッチコピーと、夕暮れの美しいビーチの背景画像。
その下には「行き先で探す」「テーマで探す」「予算で探す」という3つの特徴的な検索カードを配置。
全体の配色は青と白を基調とし、クリーンで信頼感のある印象を与える。
"""

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

# Colab上に画像を表示
display(web_design_image)
```

## 6. 広告用の商品写真を生成し、対話的に編集する

新商品の広告キャンペーンで使うような、プロフェッショナルで魅力的な画像を生成し、さらにそこから対話を通じて修正を加えていく過程を体験してみましょう。

まずは、ベースとなる飲料の広告写真を作成します。

```python
# 広告写真のプロンプト
prompt = """
新しいエナジードリンク「SPARK」の広告写真。
水滴がたくさんついた冷たいアルミ缶が、砕いた氷の上に置かれている。
背景はシャープでモダンな青いグラデーション。
缶には「SPARK」という文字がスタイリッシュなフォントで描かれている。
エネルギッシュで爽快なイメージ。
非常に高画質で、広告に使用できるレベル。
"""

# モデルを呼び出して画像を生成
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

# 結果を表示
ad_image = None
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image = Image.open(BytesIO(part.inline_data.data))
      ad_image.save('generated_drink_ad.png')
      print("生成された広告画像を 'generated_drink_ad.png' として保存しました。")

# Colab上に画像を表示
if ad_image:
    display(ad_image)
```

### 6.1. 生成した画像に人物を追加する

次に、この商品写真に人物を追加して、より魅力的な広告にしてみましょう。「この缶を人が持っているように」と指示します。

```python
# 人物を追加するプロンプト
prompt = "このドリンクの缶を、スポーティーな若い女性が笑顔で持っているように編集してください。商品の見え方や背景の雰囲気は維持してください。"

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image], # テキストプロンプトと元の画像を渡す
)

# 結果を表示
ad_image_with_person = None
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_with_person = Image.open(BytesIO(part.inline_data.data))
      ad_image_with_person.save('ad_with_person.png')
      print("人物を追加した画像を 'ad_with_person.png' として保存しました。")

# Colab上に画像を表示
if ad_image_with_person:
    display(ad_image_with_person)
```

### 6.2. 商品のロゴを変更する

急なデザイン変更にも対応できます。缶に描かれたロゴを、別の名前に変更してみましょう。

```python
# ロゴを変更するプロンプト
prompt = "缶に書かれている「SPARK」というロゴを、新しい「AQUA」というロゴに自然な形で変更してください。フォントのスタイルは元のロゴに似せてください。"

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image_with_person], # 人物追加後の画像をベースにする
)

# 結果を表示
ad_image_new_logo = None
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_new_logo = Image.open(BytesIO(part.inline_data.data))
      ad_image_new_logo.save('ad_new_logo.png')
      print("ロゴを変更した画像を 'ad_new_logo.png' として保存しました。")

# Colab上に画像を表示
if ad_image_new_logo:
    display(ad_image_new_logo)
```

### 6.3. 背景の色を調整する

最後に、商品のフレーバーに合わせて背景色を変更してみましょう。オレンジ味を想定して、背景を暖色系に変えてみます。

```python
# 背景色を変更するプロンプト
prompt = "背景全体を、暖色系のオレンジ色の鮮やかなグラデーションに変更してください。人物や商品には影響を与えないでください。"

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, ad_image_new_logo], # ロゴ変更後の画像をベースにする
)

# 結果を表示
ad_image_final = None
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      ad_image_final = Image.open(BytesIO(part.inline_data.data))
      ad_image_final.save('ad_final.png')
      print("最終版の広告画像を 'ad_final.png' として保存しました。")

# Colab上に画像を表示
if ad_image_final:
    display(ad_image_final)
```
このように、一度画像を生成してから、対話を繰り返すことで、細かな要望を反映させながら完成度を高めていくことができます。

## 7. スケッチでポーズを指示して、人物を動かす

Nano Banana の真骨頂ともいえるのが、複数の画像を組み合わせて編集する能力です。ここでは、簡単な棒人間のスケッチでポーズを指示し、実際の人物写真にそのポーズをとらせてみましょう。

まず、ポーズの指示に使うスケッチと、元になる人物の写真を準備します。

```python
# サンプルのスケッチと人物写真をダウンロード
!curl -o pose_sketch.jpg "https://storage.googleapis.com/generativeai-downloads/images/pose_sketch.png"
!curl -o person.jpg "https://storage.googleapis.com/generativeai-downloads/images/person.jpg"

# 画像を開いて表示
sketch_image = Image.open('pose_sketch.jpg')
person_image = Image.open('person.jpg')

print("指示に使うスケッチ:")
display(sketch_image)
print("元になる人物写真:")
display(person_image)
```

次に、これらの画像と「何をしてほしいか」をテキストで伝えます。

```python
# スケッチを使った画像編集のプロンプト
prompt = "この人物（右の画像）に、左のスケッチと同じ、喜びにあふれたダイナミックなジャンプのポーズをとらせてください。服装、背景、人物の顔は元の写真のスタイルを維持してください。"

# モデルを呼び出して画像を編集
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, sketch_image, person_image], # テキスト、スケッチ画像、人物画像の3つをリストで渡す
)

# 結果を表示
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      posed_image = Image.open(BytesIO(part.inline_data.data))
      posed_image.save('posed_person.png')
      print("ポーズを編集した画像を 'posed_person.png' として保存しました。")

# Colab上に画像を表示
display(posed_image)
```
このように、スケッチという曖昧な情報からでも、モデルが意図を汲み取って、写真の人物に自然なポーズをとらせることができるのです。

さあ、あなたも様々なプロンプトや手持ちの画像を使って、Nano Banana の驚くべき可能性を探求してみてください！