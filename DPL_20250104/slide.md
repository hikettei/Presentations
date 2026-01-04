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

[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)  ◀ exec                    │
│     S(0,0)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 0                              │  │   STORE(C,k,a+b)                           │
│ t = 0                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: a ← LOAD(A,k)                        │  │ A=[[>1<,2],[3,4]]                          │
│ regs: a=1  b=·                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : idle                                 │  │ C=[[0,0],[0,0]]                            │
│                                            │  │                                            │
│ S(0,0)  k=0                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 01/12  [■□□□□□□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===
``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(0,0)  ◀ now                          │  │   b = LOAD(B,k)  ◀ exec                    │
│ k = 2*i+j = 0                              │  │   STORE(C,k,a+b)                           │
│ t = 1                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: b ← LOAD(B,k)                        │  │ A=[[1,2],[3,4]]                            │
│ regs: a=1  b=4                             │  │ B=[[>4<,3],[2,1]]                          │
│ ALU : idle                                 │  │ C=[[0,0],[0,0]]                            │
│                                            │  │                                            │
│ S(0,0)  k=0                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 02/12  [■■□□□□□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(0,0)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 0                              │  │   STORE(C,k,a+b)  ◀ exec                   │
│ t = 2                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: STORE(C,k,a+b)                       │  │ A=[[1,2],[3,4]]                            │
│ regs: a=1  b=4                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : a+b=5                                │  │ C=[[>5<,0],[0,0]]                          │
│                                            │  │                                            │
│ S(0,0)  k=0                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 03/12  [■■■□□□□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)  ◀ exec                    │
│     S(0,1)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 1                              │  │   STORE(C,k,a+b)                           │
│ t = 3                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: a ← LOAD(A,k)                        │  │ A=[[1,>2<],[3,4]]                          │
│ regs: a=2  b=·                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : idle                                 │  │ C=[[5,0],[0,0]]                            │
│                                            │  │                                            │
│ S(0,1)  k=1                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 04/12  [■■■■□□□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(0,1)  ◀ now                          │  │   b = LOAD(B,k)  ◀ exec                    │
│ k = 2*i+j = 1                              │  │   STORE(C,k,a+b)                           │
│ t = 4                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: b ← LOAD(B,k)                        │  │ A=[[1,2],[3,4]]                            │
│ regs: a=2  b=3                             │  │ B=[[4,>3<],[2,1]]                          │
│ ALU : idle                                 │  │ C=[[5,0],[0,0]]                            │
│                                            │  │                                            │
│ S(0,1)  k=1                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 05/12  [■■■■■□□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(0,1)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 1                              │  │   STORE(C,k,a+b)  ◀ exec                   │
│ t = 5                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: STORE(C,k,a+b)                       │  │ A=[[1,2],[3,4]]                            │
│ regs: a=2  b=3                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : a+b=5                                │  │ C=[[5,>5<],[0,0]]                          │
│                                            │  │                                            │
│ S(0,1)  k=1                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 06/12  [■■■■■■□□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)  ◀ exec                    │
│     S(1,0)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 2                              │  │   STORE(C,k,a+b)                           │
│ t = 6                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: a ← LOAD(A,k)                        │  │ A=[[1,2],[>3<,4]]                          │
│ regs: a=3  b=·                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : idle                                 │  │ C=[[5,5],[0,0]]                            │
│                                            │  │                                            │
│ S(1,0)  k=2                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 07/12  [■■■■■■■□□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(1,0)  ◀ now                          │  │   b = LOAD(B,k)  ◀ exec                    │
│ k = 2*i+j = 2                              │  │   STORE(C,k,a+b)                           │
│ t = 7                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: b ← LOAD(B,k)                        │  │ A=[[1,2],[3,4]]                            │
│ regs: a=3  b=2                             │  │ B=[[4,3],[>2<,1]]                          │
│ ALU : idle                                 │  │ C=[[5,5],[0,0]]                            │
│                                            │  │                                            │
│ S(1,0)  k=2                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 08/12  [■■■■■■■■□□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(1,0)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 2                              │  │   STORE(C,k,a+b)  ◀ exec                   │
│ t = 8                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: STORE(C,k,a+b)                       │  │ A=[[1,2],[3,4]]                            │
│ regs: a=3  b=2                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : a+b=5                                │  │ C=[[5,5],[>5<,0]]                          │
│                                            │  │                                            │
│ S(1,0)  k=2                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 09/12  [■■■■■■■■■□□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)  ◀ exec                    │
│     S(1,1)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 3                              │  │   STORE(C,k,a+b)                           │
│ t = 9                                      │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: a ← LOAD(A,k)                        │  │ A=[[1,2],[3,>4<]]                          │
│ regs: a=4  b=·                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : idle                                 │  │ C=[[5,5],[5,0]]                            │
│                                            │  │                                            │
│ S(1,1)  k=3                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 10/12  [■■■■■■■■■■□□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(1,1)  ◀ now                          │  │   b = LOAD(B,k)  ◀ exec                    │
│ k = 2*i+j = 3                              │  │   STORE(C,k,a+b)                           │
│ t = 10                                     │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: b ← LOAD(B,k)                        │  │ A=[[1,2],[3,4]]                            │
│ regs: a=4  b=1                             │  │ B=[[4,3],[2,>1<]]                          │
│ ALU : idle                                 │  │ C=[[5,5],[5,0]]                            │
│                                            │  │                                            │
│ S(1,1)  k=3                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 11/12  [■■■■■■■■■■■□]
```
<!-- cmd:end_slide -->
[Part1] (2/N) Introduction: 行列の加算を実際に動かしてみる
===

``` python
╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ loop                                       │  │ program                                    │
│ for i in {0,1}:                            │  │ S(i,j) {                                   │
│   for j in {0,1}:                          │  │   a = LOAD(A,k)                            │
│     S(1,1)  ◀ now                          │  │   b = LOAD(B,k)                            │
│ k = 2*i+j = 3                              │  │   STORE(C,k,a+b)  ◀ exec                   │
│ t = 11                                     │  │ }                                          │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

╭────────────────────────────────────────────╮  ╭────────────────────────────────────────────╮
│ state                                      │  │ tensors (memory)                           │
│ step: STORE(C,k,a+b)                       │  │ A=[[1,2],[3,4]]                            │
│ regs: a=4  b=1                             │  │ B=[[4,3],[2,1]]                            │
│ ALU : a+b=5                                │  │ C=[[5,5],[5,>5<]]                          │
│                                            │  │                                            │
│ S(1,1)  k=3                                │  │ note: >x< = touched                        │
│                                            │  │                                            │
╰────────────────────────────────────────────╯  ╰────────────────────────────────────────────╯

frame 12/12  [■■■■■■■■■■■■]
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
