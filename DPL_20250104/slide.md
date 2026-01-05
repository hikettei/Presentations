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

[Part1] (1/N) Introduction: 配列操作
====

# 行列の要素ごとの加算を丁寧に考えてみる

Q: 2x2行列`A, B`について，`A[i, j] + B[i, j]`を計算し，その結果を`C[i, j]`に保存するプログラムを考える

<!-- cmd:pause -->

``` python
# Inputs
A = [[1, 2], [3, 4]]
B = [[4, 3], [2, 1]]
C = [[0, 0], [0, 0]]
```

<!-- cmd:pause -->

# 実装例
<!-- cmd:column_layout: [2, 2, 1] -->
<!-- cmd:column: 0 -->

### Program

```python
for i in range(2):
  for j in range(2):
    C[i, j] = A[i, j] + B[i, j]
# print(C) -> [[5, 5], [5, 5]]
````

<!-- cmd:column: 1 -->

### Schedule

```python
for i in range(2):
  for j in range(2):
    print(f"S({i}, {j})")
```

<!-- cmd:column: 2 -->

### Timestamp

```text
t=0 | S(0, 0)
t=1 | S(0, 1)
t=2 | S(1, 0)
t=3 | S(1, 1)
```

<!-- cmd:reset_layout -->

<!-- cmd:pause -->

## Pythonプログラムをよく観察してみる

- 合計で`2 * 2 = 4`回，計算を実行する `S(i, j): S(0, 0) -> S(0, 1) -> S(1, 0) -> S(1, 1)`
<!-- cmd:pause -->
- 各Statement `S(i, j)`において，こういうことをやってそう:
<!-- cmd:pause -->
  1. `A[i, j]`をメモリからレジスタaへロードする
<!-- cmd:pause -->
  2. `B[i, j]`をメモリからレジスタbへロードする
<!-- cmd:pause -->
  3. `a+b`を計算する
<!-- cmd:pause -->
  4. (3.)の実行結果を，`C[i, j]`へ保存する
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

# Data Processing in general
  
- `DATA`: 計算したいデータがある (e.g.: NN Parameter Weight, 口座残高，年齢，etc ...)
  - データ型 (e.g.: 小数点，文字列, Boolean)
<!-- cmd:pause -->
- `g(i)`: メモリからデータをどういう順番で読むか？ (e.g.: ランダムアクセス，規則的)
  - 例: `g(i, j) = 4i+j (Strided-Array)`, `g(i) = random(0, 4)` 
  - Deep Learningで用いるアルゴリズムの95%は，f(i)がQuasiaffine関数であることが知られている (TODO: SOurce)
  - (注: Quasiaffine, fがPresburger算術のclass, 要は+と*のみで表記できるaffineな関数)
<!-- cmd:pause -->
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
- TODO: N段階の並列性

<!-- cmd:end_slide -->

[Part1] (6/N) 計算機のアーキテクチャ
======

(Disclaimer: 僕はプロの半導体屋さんではありません ...)

<!-- cmd:column_layout: [2, 4] -->
<!-- cmd:column: 0 -->

計算機を構成する要素は，僕は以下の三つだと考えている。

- メモリ (storage / memory / cache)
  - サイズ: N byte
  - 帯域幅: X GB/sec
  - エネルギー a pJ/word

<!-- cmd:column: 1 -->

```python
            ╭────────────────────────────────────────────╮
            │ SSD / NVMe (storage)               ~  TB   │
            ╰────────────────────────────────────────────╯
                           ║
                           ║  (I/O)
                           ║
            ╭────────────────────────────────────────────╮
            │ LPDDR / DRAM (main memory)         ~  GB   │
            ╰────────────────────────────────────────────╯

                ╭──────────────────────────────╮
                │ On-chip SRAM (L2/LLC)  ~  MB │
                ╰──────────────────────────────╯

                     ╭──────────────────╮
                     │ Local SRAM ~ KB  │
                     │ (SMEM / REG)     │
                     ╰──────────────────╯
```

<!-- cmd:reset_layout -->
<!-- cmd:end_slide -->

[Part1] (6/N) 計算機のアーキテクチャ
======

(Disclaimer: 僕はプロの半導体屋さんではありません ...)

<!-- cmd:column_layout: [2, 4] -->
<!-- cmd:column: 0 -->

計算機を構成する要素は，僕は以下の三つだと考えている。

- メモリ (storage / memory / cache)
  - サイズ: N byte
  - 帯域幅: X GB/sec
  - エネルギー a pJ/word
- ALU

<!-- cmd:column: 1 -->

```python
            ╭────────────────────────────────────────────╮
            │ SSD / NVMe (storage)               ~  TB   │
            ╰────────────────────────────────────────────╯
                           ║
                           ║  (I/O)
                           ║
            ╭────────────────────────────────────────────╮
            │ LPDDR / DRAM (main memory)         ~  GB   │
            ╰────────────────────────────────────────────╯

                ╭──────────────────────────────╮
                │ On-chip SRAM (L2/LLC)  ~  MB │
                ╰──────────────────────────────╯

                     ╭──────────────────╮
                     │ Local SRAM ~ KB  │
                     │ (SMEM / REG)     │
                     ╰──────────────────╯
                           
                           
                        ╭──────╮
                        │ ALU  │
                        │ F(.) │
                        ╰──────╯
```

<!-- cmd:reset_layout -->
<!-- cmd:end_slide -->

[Part1] (6/N) 計算機のアーキテクチャ
======

(Disclaimer: 僕はプロの半導体屋さんではありません ...)

<!-- cmd:column_layout: [2, 4] -->
<!-- cmd:column: 0 -->

計算機を構成する要素は，僕は以下の三つだと考えている。

- メモリ (storage / memory / cache)
  - サイズ: N byte
  - 帯域幅: X GB/sec
  - エネルギー a pJ/word
- ALU
- チップ内ネットワーク

<!-- cmd:column: 1 -->

```python
            ╭────────────────────────────────────────────╮
            │ SSD / NVMe (storage)               ~  TB   │
            ╰────────────────────────────────────────────╯
                           ║
                           ║  (I/O)
                           ║
            ╭────────────────────────────────────────────╮
            │ LPDDR / DRAM (main memory)         ~  GB   │
            ╰────────────────────────────────────────────╯
                           │  640 pJ/word
                           ▼
                ╭──────────────────────────────╮
                │ On-chip SRAM (L2/LLC)  ~  MB │
                ╰──────────────────────────────╯
                           │   50 pJ/word
                           ▼
                     ╭──────────────────╮
                     │ Local SRAM ~ KB  │
                     │ (SMEM / REG)     │
                     ╰──────────────────╯
                           │   5 pJ/word
                           ▼
                        ╭──────╮
                        │ ALU  │
                        │ F(.) │
                        ╰──────╯
```

<!-- cmd:reset_layout -->
<!-- cmd:end_slide -->
[Part1] (7/N) GPUの特性1: Communication is expensive
===
TODO: 引用
<!-- cmd:end_slide -->
[Part1] (8/N) GPUの特性2: ALU <<< Memory
===
TODO: 引用
<!-- cmd:end_slide -->
[Part2] (1/N) 計算機を効率良く扱うためにはどうしたらいいか？
===

前述の通り，キャッシュなどを考慮しないとGPUはパフォーマンスが出ない

=> N=10^100とかまでデータが増えたら？
=> Gemmのような，Memory-Intensiveな演算はどうする？
=> 一回のALUを呼び出すのにSSDから10回呼び出す
=> 一回のALUを呼び出すのにL3から10回呼び出す

# throughput-oriented metrics: (TODO: 引用)

上記のスライドでは，throughputが関連する指標として以下があるとしている

# throughput

throughput = A

良くB/Fやメモリ帯域幅から理論値のFLOPSを計算して，それに近づくようにプログラムを最適化したりする

FLOPS, B/F メモリ通信とALUの性能の比率メモリが遊んでるか演算機が遊んでるか

<!-- cmd:pause -->
## Amount of data to be applied. (入力するデータの総量)
<!-- cmd:pause -->
これを減らすには，アルゴリズムを変えるか，Quantization/Pruningなどでデータ量を減らすしかない
<!-- cmd:pause -->
## Amount of resources to be applied (enegery, silicons) 
<!-- cmd:pause -->
お金をいっぱい投入するしかない
<!-- cmd:pause -->
## Efficiency of applying them to useful work
<!-- cmd:pause -->
=> これは，キャッシュなどを考慮した"良いプログラム"に書き換えることで改善できる。
=> 計算の意味を変えず，導線だけを変える"Loop Transformation"という考え方を使う
<!-- cmd:end_slide -->

[Part2] (2/N) Tile
===

- 床のタイルとかと同じ意味
- 100x100の長方形は，10x10のタイル100枚敷き詰めている

要素間の依存関係, Tile操作, Polyhedral Model

- 1000あるデータを100人で分割して同時に作業する (=重要な操作，Tileの定義)
- 集約(aggregation)する
- ここで，同時に作業する人の順番をランダムにシャッフルしても壊れない = プログラムは並列である！ Coincidenceの数学的定義
- 実際は
- 10000件あるデータを(a1, a2, ..., a100)さんで100分割して，一人当たり100件作業する = (block, thread) = (100, 100)
- (a1, a2, ..., a100)さんはそれを下請け業者(b1, b2)に業務委託する = WarpLevel
- Tile操作の考え方は，GPU Kernelを最適化する上でとても基本的な事項 (現在最も広く使われているLLM Inference Server, SGLangのバックエンドのコンパイラは"TileLang"って名前だったりする)

ここから抽出されたGPU Kernelの要素
=> メモリアクセスの依存関係 (RaW/WaW/WaR)
=> スケジュールの合法性 (legality)
=> スケジュールの並列性 (coincidence)

<!-- cmd:end_slide -->
[Part2] (3/N) 並列化 (Loop Parallelize for CPU)
===
(TODO: Polyhedral Compilerを用いて説明する)
<!-- cmd:end_slide -->

[Part2] (4/N) 並列化 (Loop Parallelize for GPU)
===

<!-- cmd:end_slide -->

[Part2] (5/N) 並列化 (Strip-Mine, SIMD)
===
TensorCore: 4x4 TileとかでA@B=Cを計算する
<!-- cmd:end_slide -->
[Part2] (6/N) Memory Locality効率化 (Loop Coalesce)
===
<!-- cmd:end_slide -->
[Part2] (7/N) Memory Locality効率化 (Tiling)
===
<!-- cmd:end_slide -->
[Part2] (8/N) Memory Locality効率化 (Interchange)
===
<!-- cmd:end_slide -->
[Part3] (9/N) Memory Locality効率化 (Loop Fusion)
===
- Loop Fusion (TODO: 根拠の論文を持ってくる) Which is NP-Hard problem to optimize.
  - 応用: On-the-fly reduction, FlashAttention (ざっくり言えば，Matmul+Softmax+Matmulを全てLoop Fusionした形として説明できる，Softmax安定化のコード変形に目を瞑れば)
<!-- cmd:end_slide -->
[Part3] (10/N) Memory Locality効率化 (Loop Skewing)
===
- Stencil/Skewing (NxMの領域を三角形のタイルで埋めていく，論文どこいったっけ)
<!-- cmd:end_slide -->

[Part3] (1/N) Deep Learning Compiler
======

(Disclaimer: この部分は本当にいろんなアプローチがあります。Halide, Tiramisu, Polyhedral Model, Tinygrad, E-Graph and Equality Saturation, etc ...)

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

次に読むと面白いかもしれない文献
CUDAで最高速度のGemmを書くBlog
<!-- cmd:end_slide -->
[Part3] (2/N) Conclusion? HalideでGemmを書いてみる
===

<!-- cmd:end_slide -->

参考文献
======
- https://microarch.org/micro52/media/dally_keynote.pdf
- https://www.slideshare.net/slideshow/introduction-to-polyhedral-compilation/70482946
- https://pliss2019.github.io/albert_cohen_slides.pdf

<!-- cmd:end_slide -->
