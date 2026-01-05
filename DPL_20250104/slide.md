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

[Part1] (1/N) Introduction: 行列演算
====

# 要素ごとの加算 (blas_saxpy where a = 1.0)

Q: 2x2行列`A, B`について，`A[i, j] + B[i, j]`を計算し，その結果を`C[i, j]`に保存するプログラムを考える

<!-- cmd:pause -->

``` python
# Inputs
A = [[1, 2], [3, 4]] # float32
B = [[4, 3], [2, 1]] # float32
C = [[0, 0], [0, 0]] # float32
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

## プログラムをよく観察してみる

- 合計で`2 * 2 = 4`回，計算を実行する `S(i, j): S(0, 0) -> S(0, 1) -> S(1, 0) -> S(1, 1)`
<!-- cmd:pause -->
- 各Statement `S(i, j)`において，こういうことをやってそう:
<!-- cmd:pause -->
  1. `A[i, j]`, `B[i, j]`をメモリからレジスタa/bへロードする (16byte load)
<!-- cmd:pause -->
  2. `a+b`を計算する (1 FLOP)
<!-- cmd:pause -->
  3. (3.)の実行結果を，`C[i, j]`へ保存する (8byte store)
<!-- cmd:end_slide -->
[Part1] (2/N) 計算機モデル (Memory/ALU)
===

```python
╭Memory────╮  ╭Memory────╮  ╭Memory────╮
│ f =  +   │  │A[i,j] = 1│  │B[i,j] = 4│
╰────┬─────╯  ╰────┬─────╯  ╰────┬─────╯
     │ FETCH inst  │ LOAD 8B     │ LOAD 8B    (total LOAD = 16B)
     │             │             │
     │             │             │
╭ALU─┼─────────────┼─────────────┼──────╮
│    ▼             ▼             ▼      │
│    f             a             b      │     (total FLOP = 1)
│ out = f(a,b) = 5             (1 FLOP) │
╰──┼────────────────────────────────────╯
   │ STORE 8B                               
   │                   
   │        ╭Memory────╮                      (total STORE = 8B)
   └───────▶︎│C[i,j] = 5│
            ╰──────────╯            
(B/F = (LOAD+STORE)/FLOP = 24)
```

計算機を極言すると，以下の三つをするだけのマシン:
<!-- cmd:pause -->
1. **LOAD**: データをメモリから持ってくる。(`total=size_of(float)*n_element*2`)
<!-- cmd:pause -->
2. **ALU**: レジスタ上で命令を演算する。 (1 FLOP)
<!-- cmd:pause -->
3. **STORE**: 結果をメモリへ書き戻す。(`total=size_of(float)*n_element*2`)
<!-- cmd:pause -->
## FLOP
- 一回の浮動小数点演算の単位
<!-- cmd:pause -->
## B/F (Bytes per FLOP)
- B/F メモリ性能 vs 演算性能

<!-- cmd:end_slide -->

[Part1] (3/N) Introduction: データ処理
====

``` python
╭────────────── ALU ───────────────────╮
│ y = f( DATA1[ g(i) ] , DATA2[ g(j) ])│
╰───▲───────────▲───────────▲──────────╯
    │           │           │
    │           │           └─ data2 (tensor / memory), accessed at g(j)
    │           └───────────── data1 (tensor / memory), accessed at g(i)
    └───────────────────────── f : algorithm to apply
```

``` python
╭─────────────── memory ──────────────────╮
│ Addr : 0   1   2   3   4   5   …        │
│ Val  : x0  x1  x2  x3  x4  x5  …        │
╰─────────────────────────────────────────╯
               ▲
            k = g(i)
```

<!-- cmd:pause -->
# Data processing in general
<!-- cmd:pause -->
- `DATA`: 計算したいデータがある (e.g.: NN Parameter Weight, 口座残高，年齢，etc ...)
  - データ型: float32 (8byte), int64 (16byte)
  - データ量: [M, N]行列
  - 総通信量: `データ型 * データ量`: `(M*N*size_of(float32))bytes` (e.g.: float32型のMN行列)
<!-- cmd:pause -->
- `g(i)`: メモリからデータをどういう順番で読むか？ (e.g.: ランダムアクセス，規則的)
  - 例: `g(i, j) = 4i+j (Strided-Array)`, `g(i) = random(0, 4)` 
  - Deep Learningで用いるアルゴリズムの95%は，f(i)がQuasiaffine関数であることが知られている (TODO: SOurce)
  - (注: Quasiaffine, fがPresburger算術のclass, 要は+と*のみで表記できるaffineな関数)
<!-- cmd:pause -->
- `f`: 読んだデータに対してどういう処理をするか？(e.g.: `+`, `*`, `replace`) (1 FLOP)

<!-- cmd:end_slide -->

[Part1] (4/N) Introduction: 深層学習におけるデータ処理
====

前述のモデルでいうと: 深層学習の計算は超規則的で簡単である

深層学習でよくやる計算

- Gemm
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

[Part1] (5/N) 計算機のアーキテクチャ
====

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

[Part1] (5/N) 計算機のアーキテクチャ
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

[Part1] (5/N) 計算機のアーキテクチャ
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

- (Ref: https://microarch.org/micro52/media/dally_keynote.pdf)

<!-- cmd:reset_layout -->
<!-- cmd:end_slide -->
[Part1] (6/N) Introduction: CPU/GPU
===
前述のスライドを持ってきた上で:
- CPU: 2段階での並列性
- GPU: n段階での並列性，VRAM/SRAM

<!-- cmd:end_slide -->
[Part1] (7/N) 計算コスト << 通信コスト
===

```python
╭───────────────────────────╮  ╭───────────────────────────╮  ╭───────────────────────────╮
│          Integer          │  │            FP             │  │        Memory (64bit)     │
├───────────────────────────┤  ├───────────────────────────┤  ├───────────────────────────┤
│ Add                       │  │ FAdd                      │  │ Cache                     │
│   8  bit   0.03 pJ        │  │   16 bit   0.4 pJ         │  │   8KB      10  pJ         │
│   32 bit   0.1  pJ        │  │   32 bit   0.9 pJ         │  │   32KB     20  pJ         │
│                           │  │                           │  │   1MB      100 pJ         │
│ Mult                      │  │ FMult                     │  │ DRAM    1.3–2.6 nJ        │
│   8  bit   0.2  pJ        │  │   16 bit   1   pJ         │  │                           │
│   32 bit   3    pJ        │  │   32 bit   4   pJ         │  │                           │
╰───────────────────────────╯  ╰───────────────────────────╯  ╰───────────────────────────╯


Instruction Energy Breakdown (example: Add)    total ≈ 70 pJ
╭──────────────────────────────────────────────────────────────────────────────╮
│ I-Cache Access 25pJ │ RegFile Access 6pJ │   Control (rest) ≈ 39pJ   │  Add  │
╰──────────────────────────────────────────────────────────────────────────────╯
        ↑ I-Cache Access         ↑ Register File Access                      ↑ Add
```

- Figures/Numbers are from Mark Horowitz “Computing’s Energy Problem (and what we can do about it)”, ISSCC 2014.

<!-- cmd:end_slide -->
[Part2] (1/N) 計算機を効率良く扱うためにはどうしたらいいか？
===

## FLOP
## B/F (Bytes per FLOPS)

``` python
for i in range(N):
  a, b = A[i], B[i] # 16 byte load
  tmp = a + b       # 1  FLOP
  out[i] = tmp      # 8  byte store
```

### プログラムの要求 B/F

- B/F = 24/1 = 24

### ハードウェアのB/F

- 高々0.5とか？

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
[Part3] (2/N) Conclusion? TinygradでMetal Gemmを書いてみる
===
Tinykitten
<!-- cmd:end_slide -->

参考文献
======
- https://microarch.org/micro52/media/dally_keynote.pdf
- https://www.slideshare.net/slideshow/introduction-to-polyhedral-compilation/70482946
- https://pliss2019.github.io/albert_cohen_slides.pdf

<!-- cmd:end_slide -->
