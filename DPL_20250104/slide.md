---
title: "High Performance Computing for Deep Learning (行列演算ポエム)"
sub_title: "川島研究会 発表資料"
authors:
  - "小田 悠真 (慶應義塾大学 環境情報学部 B1)"
date: "2026-01-06"
theme:
  path: ./../hikettei_style.yaml
options:
  command_prefix: "cmd:"
---

発表構成
======

- 合計持ち時間: 30分
  - Part1: **10分** Introduction: 深層学習のための大規模データ処理
  - Part2: **10分** Background: 計算機を効率良く使うにはどうしたらいいか
  - Part3: **10分** Advanced: Deep Learning Compiler
- 実際にコード動かして遊びたい人へ: https://github.com/hikettei/tiny_polyhedral_compiler/blob/main/examples/polyhedral_compiler.ipynb

<!-- cmd:end_slide -->

[Part1] (1/N) Introduction: 配列同士の加算
====

# Q: 行列積を計算してみよう

Note: やっぱ単純なVector addにする -> 次スライドでアニメーションでどんな感じで実行されているかをみせる，という構成でいいや

```
A @ B = C # TODO: Use LaTeX
```

``` python
import numpy as np
N, K, M = 4, 4, 4
A = np.random.randn(N, K)
B = np.random.randn(K, M)
C = np.zeros((N, M),)
# A @ B = C ?
```

<!-- cmd:pause -->

### 解答例

``` python
for i in range(N):
  for j in range(M):
    for k in range(K):
      C[i*N + j] += A[i*N + k] + B[k*N + j]
```

## Pythonプログラムをよく観察してみる

- 行列を計算する`add`関数というプログラムには:
  - 行列を保存するためのメモリと
  - 行列をどういった順番で読むかと
  - `+`演算を実行するためのALUがあるはず 
  - が概念として存在する

<!-- cmd:end_slide -->

[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
(TODO: Super elegant figure)
Animationに，S1の座標，メモリの状態，ALUみたいなのを逐一保存してみる
画面下部にYoutube seekbarみたいなの作って，tが進んでいるというのをわかりやすく
```

<!-- cmd:end_slide -->

[Part1] (3/N) Introduction: データ処理
====

``` python
╭────────────── algorithm ─────────────╮
│ y = f( DATA1[ g(i) ] , DATA2[ g(j) ])│
╰───▲───────────▲───────────▲──────────╯
    │           │           │
    │           │           └─ data2 (tensor / memory), accessed at g(j)
    │           └───────────── data1 (tensor / memory), accessed at g(i)
    └───────────────────────── f : algorithm to apply
```

``` python
[DATA]
╭──────────── memory / tensor ────────────╮
│ Addr : 0   1   2   3   4   5   …        │
│ Val  : x0  x1  x2  x3  x4  x5  …        │
╰─────────────────────────────────────────╯
               ▲
            k = g(i)
```

- Data Processing in general:
  - `DATA`: 計算したいデータがある (e.g.: NN Parameter Weight, 口座残高，年齢，etc ...)
    - データ型 (e.g.: 小数点，文字列, Boolean)
  - `g(i)`: メモリからデータをどういう順番で読むか？ (e.g.: ランダムアクセス，規則的)
    - 例: `g(i, j) = 4i+j (Strided-Array)`, `g(i) = random(0, 4)` 
    - Deep Learningで用いるアルゴリズムの95%は，f(i)がQuasiaffine関数であることが知られている (TODO: SOurce)
    - (注: Quasiaffine, fがPresburger算術のclass, 要は+と*のみで表記できるaffineな関数)
  - `f`: 読んだデータに対してどういう処理をするか？(e.g.: `+`, `*`, `replace`)

<!-- cmd:end_slide -->

[Part1] (4/N) Introduction: 深層学習におけるデータ処理
====

深層学習でよくやる計算

- Conv2D (Einsum Definition)
- Pool2d (Einsum Definition)
- FlashAttention (Einsum Definition)

(TODO: テーブル形式にする？)
データ型，メモリアクセス，アルゴリズム，Offline/Onlineくらいの違いのテーブル

- データ処理 everywhere

## Data Processing in Deep Learning

多分，SQLやTransactionより，ずっと単純なデータ処理を考えていると思う

- 流れるデータ量は，事前にわかっている (Offine Optimization)
- Deep Learningの場合，メモリアクセスパターンはとっても単純
- メモリアクセス: Elementwise, Broadcast, ReduceOpsしかない (c.f.: NCCL Docs)
- WMMA (積和演算)を高速化するかばっかり考えている
- => いい感じの数学モデルを作れそう！(後で定義する)
- (余談) MLIRを用いたTransaction Compilerなんかも実際にある https://www.lingo-db.com/

<!-- cmd:end_slide -->

[Part1] (5/N) Introduction: CPU/GPU

適当に図を持ってくる
- Apple M3, Intel, AMD


<!-- cmd:end_slide -->

[Part1] (6/N) 計算機のアーキテクチャ
======

(Disclaimer: 僕はプロの半導体屋さんではありません！)

計算機を構成する要素は，僕は以下の三つだと考えている。

(TODO: 左ペインにAAを表示)

- メモリ (VRAM/SRAM, ...)
  - サイズN byte, 通信速度 X/s, 消費電力 aW

<!-- cmd:pause -->

- ALU (演算装置)

<!-- cmd:pause -->

- チップ内ネットワーク

<!-- cmd:end_slide -->

[Part2] (1/N) 計算機を効率良く扱うためにはどうしたらいいか？
===

- throughput-oriented metrics:
  - throughputの構成要素:
    - Amount of data to be applied
      - これを減らすには，アルゴリズムを変えるか，Quantization/Pruningなどでデータ量を減らすしかない
    - Amount of resources to be applied (enegery, silicons) 
    - Efficiency of applying them to useful work 
  - FLOPS, B/F メモリ通信とALUの性能の比率メモリが遊んでるか演算機が遊んでるか

<!-- cmd:end_slide -->

[Part2] (2/N) Perf/W 
===

- Communication is Expensive, Be Small, Be Local
- ALU vs メモリ通信の消費電力のグラフ
- メモリ階層のHierarchy
- 転送速度の違い
- チップネットワーク (NCCL Docs, Broadcast, Reduce)

<!-- cmd:end_slide -->

[Part2] (3/N) Tile
===

要素間の依存関係, Tile操作, Polyhedral Model

- 1000あるデータを100人で分割して同時に作業する (=重要な操作，Tileの定義)
- 集約(aggregation)する
- ここで，同時に作業する人の順番をランダムにシャッフルしても壊れない = プログラムは並列である！ Coincidenceの数学的定義
- 実際は
- 10000件あるデータを(a1, a2, ..., a100)さんで100分割して，一人当たり100件作業する = (block, thread) = (100, 100)
- (a1, a2, ..., a100)さんはそれを下請け業者(b1, b2)に業務委託する = WarpLevel
- Sections
  - Parallelize
  - SIMD (Single Instruction Multiple Data)
  - Thread/Block Level Parallelism
- Tile操作の考え方は，GPU Kernelを最適化する上でとても基本的な事項 (現在最も広く使われているLLM Inference Server, SGLangのバックエンドのコンパイラは"TileLang"って名前だったりする)
- Stencil/Skewing (NxMの領域を三角形のタイルで埋めていく，論文どこいったっけ？)

ここから抽出されたGPU Kernelの要素
=> メモリアクセスの依存関係 (RaW/WaW/WaR)
=> スケジュールの合法性 (legality)
=> スケジュールの並列性 (coincidence)

(Note: 以下に主要なループ変形を列挙して説明する)
- Loop Interchange
- Parallelize
  - (TODO: Polyhedral Modelで図を作成する)
  - 合法である条件
- TileGPU
  - GPU Block/Thread
- SIMD (Strip-Mine)
- Loop Coalesce
- Tile
- Loop Fusion (TODO: 根拠の論文を持ってくる) Which is NP-Hard problem to optimize.
  - 応用: On-the-fly reduction, FlashAttention (ざっくり言えば，Matmul+Softmax+Matmulを全てLoop Fusionした形として説明できる，Softmax安定化のコード変形に目を瞑れば)

<!-- cmd:end_slide -->
[Part3] (1/N) Deep Learning Compiler
======

(Disclaimer: There's several approach e.g.: Halide, Tiramisu, Polyhedral Model, MLIR, E-Graph/Equality Saturation, etc...)

- 計算機を効率良く扱うには，二つのアプローチがある。
  - ハードウェア側を最適化する (クロック数を上げる，プロセスルール微細化，Systolic Array, ...)
  - ソフトウェア側を最適化する (前述の最適化をうまく使ったコードを生成する)
- 自分はソフトウェア側を最適化したいと思った。
- Compiler:
``` python

unoptimized code -> [compiler] -> optimized code
    ↑                                  ↑
    ------------------------------------
     Problem: この過程で，コードが正しいことをどうやって保証するか？
```
<!-- cmd:end_slide -->

[Part3] (2/N) Schedule and Algoritm Separation, DSL
===

- よくないやり方？:
  - 一つのプログラミング言語で，計算の意味と最適化を両方一気にやる 
- 二つのプログラミング言語に分割する:
  - 計算の意味を記述する言語 (e.g.: AとBの行列積を取って，sigmoid関数を適用して。。。)
  - 上記のプログラムを最適化する言語 (e.g.: 一つ目のループを並列化して，次のループをタイルして，...)
- Example
- 先行研究: BEAM Search, Ansor, AutoTVM, Tinygrad, Luminal, XLA, 

参考文献
======
- https://microarch.org/micro52/media/dally_keynote.pdf
- https://www.slideshare.net/slideshow/introduction-to-polyhedral-compilation/70482946
- https://pliss2019.github.io/albert_cohen_slides.pdf

<!-- cmd:end_slide -->
