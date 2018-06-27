# Slug machine

Slug machineは計算モデルの一つです。
メモリを表すテープとテープを読み取るヘッダ、各ステップに渡って状態を保持するレジスタからなります。
テープは、左の端はありますが、右には無限に長いベクトル列です。ここで言うベクトルは固定長のバイナリ列です。
ヘッダは、テープの左端の位置から始まり、各ステップごとに右に１つずつ動きます。各ステップでヘッダはヘッダ位置のテープのベクトル(チャンクと呼びます)を読み取り、レジスタの更新と同時にヘッダ位置のベクトルも更新し、右へ動きます。もし、ヘッダ位置のベクトルが0ベクトルならテープの左端へ戻ります。
レジスタは固定長の実数列です。

# 実行
```
$ python main.py trace_test.txt --tape_width=8 --state_size=8
```

```
$ python main.py trace_helloworld.txt --tape_width=8 --state_size=32 --epoch=1000
```
