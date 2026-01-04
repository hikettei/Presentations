---
title: "Introduction to High Performance Computing (行列演算ポエム)"
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
- サンプルコードはここに置いています: https://github.com/hikettei/tiny_polyhedral_compiler/blob/main/examples/polyhedral_compiler.ipynb

<!-- cmd:end_slide -->

[Part1] (1/N) 行列演算の重要性
=======

大規模なデータを扱うことは重要！

(Note: Attentionの回もそうだけど，なぜこれが重要かという話を最初に持ってくる)

これまでのデータベースのような話と同じで

Deep Learningは，巨大なデータがあって，それに対する処理があって，それを高速化するというのを考える。

Transactionとかと話は似ている

(余談) MLIRを用いたTransaction Compilerなんかも実際にある https://www.lingo-db.com/

<!-- cmd:end_slide -->

[Part1] (2/N) 計算機のアーキテクチャ
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

(ここで，　GPU全体像の図を作る)

CPUからデータを持ってくる，CUDAはGPU launch overheadがある
Grid/Block Levelでの並列化
Warp levelでの並列化

<!-- cmd:end_slide -->

[Part2] (1/N) 計算機を効率良く扱うにはどうしたらいいか
===

- FLOPS, B/F メモリ通信とALUの性能の比率メモリが遊んでるか演算機が遊んでるか
- ALU vs メモリ通信の消費電力のグラフ
- メモリ階層のHierarchy
- 転送速度の違い

<!-- cmd:end_slide -->

GPUをどうやって効率的に扱う？
===

Tile操作, Polyhedral Model

- 1000あるデータを100人で分割して同時に作業する (=重要な操作，Tileの定義)
- 集約(aggregation)する
- ここで，同時に作業する人の順番をランダムにシャッフルしても壊れない = プログラムは並列である！ Coincidenceの数学的定義
- 実際は
- 10000件あるデータを(a1, a2, ..., a100)さんで100分割して，一人当たり100件作業する = (block, thread) = (100, 100)
- (a1, a2, ..., a100)さんはそれを下請け業者(b1, b2)に業務委託する = WarpLevel
- Parallel
- SIMD
- Thread/Block Level Parallel

References
======
- https://microarch.org/micro52/media/dally_keynote.pdf
<!-- cmd:end_slide -->
