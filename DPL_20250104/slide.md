---
title: "High Performance Computing for Deep Learning (行列演算ポエム)"
sub_title: "川島研究会 発表資料"
authors:
  - "小田 悠真 (環境情報B1)"
date: "2026-01-06"
theme:
  path: ./../hikettei_style.yaml
options:
  command_prefix: "cmd:"
---

お品書き
======

- 合計持ち時間: 30分
  - Part1: **10分** 前提用語の整理・高性能計算 推論チップの構成など (Background)
  - Part2: **10分** 計算機をうまく使うにはどうしたらいいかの話
  - Part3: **10分** Deep Learning Compiler (Halide or Polyhedral Compiler)
- 実際にコード動かして遊びたい人へ: https://github.com/hikettei/tiny_polyhedral_compiler/blob/main/examples/polyhedral_compiler.ipynb

<!-- cmd:end_slide -->

[Part1] (1/N) Introduction: 深層学習のための大規模データ処理
====

- DISK[f(i)]
- ディスクから，`f(i)`番目のデータを取ってくる
- Note
  - https://www.slideshare.net/slideshow/introduction-to-polyhedral-compilation/70482946
  - https://pliss2019.github.io/albert_cohen_slides.pdf
- **DATA** to be applied
- **Algorithm** to apply
(TODO: 話す内容を整理する，AAを用いて，処理したいデータ -> 処理 -> 処理されたデータのWorkflowをはっきりさせる)

- ここで存在するものは何か:
  - データ: リスト，Contiguous Array, Random Access? Affine Access? (TODO: Affine Accessである根拠のスライドを持ってくる)
  - 処理で行う計算は，MLIRのような中間表現を用いて処理する。

大規模なデータを扱うことは重要！

(Note: Attentionの回もそうだけど，なぜ現代でHPCが重要な技術とみなされているかをaudienceに納得させてから話したい)

これまでのデータベースのような話と同じで

Deep Learningは，巨大なデータがあって，それに対する処理があって，それを高速化するというのを考える。

Transactionとかと話は似ている

Deep Learningで実施する大規模データ処理の問題設定をはっきりさせる せっかくならこれまでのスライドと親和性を持たせる

(余談) MLIRを用いたTransaction Compilerなんかも実際にある https://www.lingo-db.com/

<!-- cmd:end_slide -->

[Part1] (2/N) 計算機
======

(Disclaimer: 僕はプロの半導体屋さんではありません!)

``` python
A = [1, 2, 3, 4]
B = [1, 2, 3, 4]
out = A + B
```

計算機を構成する要素は，僕は以下の三つだと考えている。

(TODO: 左ペインにAAを表示)

- メモリ (VRAM/SRAM, ...)

<!-- cmd:pause -->

- ALU (演算装置)

<!-- cmd:pause -->

- チップ内ネットワーク

<!-- cmd:end_slide -->

[Part1] (3/N) GPUがどうなってるか
======

(ここで，　GPU全体像の図を作る？わんちゃん飛ばしたほうがわかりやすいかも)

CPUからデータを持ってくる，CUDAはGPU launch overheadがある
Grid/Block Levelでの並列化
Warp levelでの並列化

<!-- cmd:end_slide -->

[Part2] (1/N) 計算機を効率良く扱うにはどうしたらいいか
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
- 二つのプログラミング言語に分割する:
  - 実行したい計算を数学的に記述するためのプログラミング言語
  - ↑のプログラムを，最適化するためのプログラミング言語
- Example
- 先行研究: BEAM Search, Ansor, AutoTVM, Tinygrad, Luminal, XLA, 


Awknoledgements
======
- https://microarch.org/micro52/media/dally_keynote.pdf
<!-- cmd:end_slide -->
